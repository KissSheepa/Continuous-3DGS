#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import imageio
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON
import random


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        fid = int(image_name) / (num_frames - 1)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame['time']

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :,
                                                  3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(
                np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[
                                            0],
                                        height=image.size[1], fid=frame_time))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
#     # print("Reading Training Transforms")
    
#     # # 1. 读取所有的训练数据 (原始全量数据)
#     # full_train_cam_infos = readCamerasFromTransforms(
#     #     path, "transforms_train.json", white_background, extension)
    
#     # # ------------------------------------------------------------------
#     # # [方案一实现] 固定间隔稀疏采样 (Fixed Interval Sparse Sampling)
#     # # ------------------------------------------------------------------
    
#     # # 设定采样间隔，例如每隔 4 帧取 1 帧 (相当于保留 25% 的数据)
#     # # 如果原来的 fps 是 30，现在就相当于 fps = 7.5
#     # interval = 4 
    
#     # # 1. 构建训练集索引：0, 4, 8, 12...
#     # # 这样能保证时间覆盖范围不变，但密度大大降低
#     # train_indices = list(range(0, len(full_train_cam_infos), interval))
    
#     # # 2. 构建测试集索引：那些被跳过的中间帧
#     # # 例如：1, 2, 3, 5, 6, 7...
#     # # 这些帧对于模型来说是“未见过的”，最能考验插值能力
#     # test_indices = [i for i in range(len(full_train_cam_infos)) if i not in train_indices]
    
#     # # 分割数据
#     # train_cam_infos = [full_train_cam_infos[i] for i in train_indices]
    
#     # # 将被丢弃的训练帧作为本次实验的测试集 (Held-out set)
#     # held_out_test_infos = [full_train_cam_infos[i] for i in test_indices]
    
#     # print(f"--- Sparse Sampling Experiment (Interval={interval}) ---")
#     # print(f"Total frames: {len(full_train_cam_infos)}")
#     # print(f"Sparse Training frames: {len(train_cam_infos)}")
#     # print(f"Held-out Test frames: {len(held_out_test_infos)}")
    
#     # # ------------------------------------------------------------------

#     # print("Reading Test Transforms")
#     # # 原来的测试集依然读取，但不参与训练
#     # original_test_cam_infos = readCamerasFromTransforms(
#     #     path, "transforms_test.json", white_background, extension)

#     # # [重要] 在稀疏实验中，绝对不能把 original_test_cam_infos 加到训练集里
#     # if not eval:
#     #     pass 
#     #     # train_cam_infos.extend(test_cam_infos) # 注释掉这行！
#     #     # test_cam_infos = []
    
#     # # 构建最终用于评估的测试集
#     # # 这里我们主要关心模型在"填补空缺"方面的能力，所以用 held_out_test_infos
#     # final_test_cameras = held_out_test_infos 

#     # nerf_normalization = getNerfppNorm(train_cam_infos)   

#     # print("Reading Training Transforms")
#     # train_cam_infos = readCamerasFromTransforms(
#     #     path, "transforms_train.json", white_background, extension)
#     # print("Reading Test Transforms")
#     # test_cam_infos = readCamerasFromTransforms(
#     #     path, "transforms_test.json", white_background, extension)

#     # if not eval:
#     #     train_cam_infos.extend(test_cam_infos)
#     #     test_cam_infos = []

#     # nerf_normalization = getNerfppNorm(train_cam_infos)

#     print("Reading Training Transforms")
    
#     # 1. 读取所有的训练数据
#     full_train_cam_infos = readCamerasFromTransforms(
#         path, "transforms_train.json", white_background, extension)
    
#     # ------------------------------------------------------------------
#     # [核心修改] 模拟不规则采样 (Irregular Sampling Simulation)
#     # ------------------------------------------------------------------
    
#     # 设定保留比例，比如只保留 20% 的帧作为训练，剩下 80% 作为“未见过的中间帧”进行测试
#     keep_ratio = 0.7
    
#     # 获取总帧数
#     total_frames = len(full_train_cam_infos)
    
#     # 随机选取索引 (保持时间顺序)
#     # 这里的 sorted 保证了时间依然是向前流动的，但间隔变得不规则了
#     indices = list(range(total_frames))
    
#     # 方案一：完全随机抽样 (最不规则)
#     # train_indices = sorted(random.sample(indices, int(total_frames * keep_ratio)))
    
#     # 方案二：大间隔抽样 (模拟低帧率)
#     # 例如每隔 5 帧取 1 帧
#     # train_indices = indices[::5] 
    
#     # 我们采用方案一来实现“不规则”
#     # 为了实验可复现，建议设置随机种子
#     random.seed(42) 
#     train_indices = sorted(random.sample(indices, int(total_frames * keep_ratio)))
    
#     # 分割数据
#     train_cam_infos = [full_train_cam_infos[i] for i in train_indices]
    
#     # 关键点：把被丢弃的训练帧，拿来做测试！
#     # 这些帧正好位于训练帧的“空隙”中，最能测试插值能力。
#     held_out_test_infos = [full_train_cam_infos[i] for i in indices if i not in train_indices]
    
#     print(f"Original training frames: {total_frames}")
#     print(f"Irregular training frames (kept): {len(train_cam_infos)}")
#     print(f"Held-out test frames (interpolated): {len(held_out_test_infos)}")
    
#     # ------------------------------------------------------------------

#     print("Reading Test Transforms")
#     # 原来的测试集依然读取，作为标准参考
#     original_test_cam_infos = readCamerasFromTransforms(
#         path, "transforms_test.json", white_background, extension)

#     if not eval:
#         # 如果不是 eval 模式 (通常指全量训练模式)，原本代码是把测试集也加进去训练
#         # 但在做这个“稀疏/不规则”实验时，绝对不能把测试集加进去！
#         # 所以这里我们保持 train_cam_infos 纯净
#         pass 
#         # train_cam_infos.extend(test_cam_infos) # 这行要注释掉或者改逻辑
#         # test_cam_infos = []
    
#     # 构建最终的测试集
#     # 你可以选择只用 held_out_test_infos (强力验证插值)
#     # 也可以把 original_test_cam_infos 加进来 (验证外推/泛化)
#     # 建议主要看 held_out_test_infos 的指标
#     final_test_cameras = held_out_test_infos 

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     ply_path = os.path.join(path, "points3d.ply")
#     if not os.path.exists(ply_path):
#         # Since this data set has no colmap data, we start with random points
#         num_pts = 100_000
#         print(f"Generating random point cloud ({num_pts})...")

#         # We create random points inside the bounds of the synthetic Blender scenes
#         xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
#         shs = np.random.random((num_pts, 3)) / 255.0
#         pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
#             shs), normals=np.zeros((num_pts, 3)))

#         storePly(ply_path, xyz, SH2RGB(shs) * 255)
#     try:
#         pcd = fetchPly(ply_path)
#     except:
#         pcd = None

#     # scene_info = SceneInfo(point_cloud=pcd,
#     #                        train_cameras=train_cam_infos,
#     #                        test_cameras=test_cam_infos,
#     #                        nerf_normalization=nerf_normalization,
#     #                        ply_path=ply_path)
#     # return scene_info
#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=final_test_cameras, # 使用我们构建的测试集
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)
#     return scene_info
#     # scene_info = SceneInfo(point_cloud=pcd,
#     #                        train_cameras=train_cam_infos,
#     #                        test_cameras=final_test_cameras, # 使用 held-out 测试集
#     #                        nerf_normalization=nerf_normalization,
#     #                        ply_path=ply_path)
#     # return scene_info


def readDTUCameras(path, render_camera, object_camera):
    camera_dict = np.load(os.path.join(path, render_camera))
    images_lis = sorted(glob(os.path.join(path, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(path, 'mask/*.png')))
    n_images = len(images_lis)
    cam_infos = []
    cam_idx = 0
    for idx in range(0, n_images):
        image_path = images_lis[idx]
        image = np.array(Image.open(image_path))
        mask = np.array(imageio.imread(masks_lis[idx])) / 255.0
        image = Image.fromarray((image * mask).astype(np.uint8))
        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        fid = camera_dict['fid_%d' % idx] / (n_images / 12 - 1)
        image_name = Path(image_path).stem
        scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]

        K, pose = load_K_Rt_from_P(None, P)
        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, -c, -b, pose[3:, :]], 0)

        S = np.eye(3)
        S[1, 1] = -1
        S[2, 2] = -1
        pose[1, 3] = -pose[1, 3]
        pose[2, 3] = -pose[2, 3]
        pose[:3, :3] = S @ pose[:3, :3] @ S

        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, c, b, pose[3:, :]], 0)

        pose[:, 3] *= 0.5

        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        FovY = focal2fov(K[0, 0], image.size[1])
        FovX = focal2fov(K[0, 0], image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readNeuSDTUInfo(path, render_camera, object_camera):
    print("Reading DTU Info")
    train_cam_infos = readDTUCameras(path, render_camera, object_camera)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfiesCameras(path):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('NeRF'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        train_img = dataset_json['ids'][::4]
        all_img = train_img
        ratio = 0.5

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    for idx in range(len(all_img)):
        image_path = all_img[idx]
        image = np.array(Image.open(image_path))
        image = Image.fromarray((image).astype(np.uint8))
        image_name = Path(image_path).stem

        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        fid = all_time[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale


# def readNerfiesInfo(path, eval):
#     print("Reading Nerfies Info")
#     cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path)

#     if eval:
#         train_cam_infos = cam_infos[:train_num]
#         test_cam_infos = cam_infos[train_num:]
#     else:
#         train_cam_infos = cam_infos
#         test_cam_infos = []

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     ply_path = os.path.join(path, "points3d.ply")
#     if not os.path.exists(ply_path):
#         print(f"Generating point cloud from nerfies...")

#         xyz = np.load(os.path.join(path, "points.npy"))
#         xyz = (xyz - scene_center) * scene_scale
#         num_pts = xyz.shape[0]
#         shs = np.random.random((num_pts, 3)) / 255.0
#         pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
#             shs), normals=np.zeros((num_pts, 3)))

#         storePly(ply_path, xyz, SH2RGB(shs) * 255)
#     try:
#         pcd = fetchPly(ply_path)
#     except:
#         pcd = None

#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)
#     return scene_info


def readNerfiesInfo(path, eval):
    print("Reading Nerfies Info")
    # 读取所有相机信息
    cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path)

    # 获取原始划分中的训练集部分作为基础数据池
    # 这里的 source_train_cam_infos 代表了原始完整的时间序列
    source_train_cam_infos = cam_infos[:train_num]

    # ------------------------------------------------------------------
    # [核心修改] 模拟不规则采样 (Irregular Sampling Simulation)
    # 逻辑与合成数据集保持完全一致
    # ------------------------------------------------------------------
    
    # 设定保留比例，保留 70% 的帧作为训练，剩下 30% 作为“未见过的中间帧”进行测试
    keep_ratio = 0.3
    
    # 获取原始训练序列的总帧数
    total_frames = len(source_train_cam_infos)
    
    # 生成索引列表
    indices = list(range(total_frames))
    
    # 设置随机种子，保证实验可复现
    random.seed(42) 
    
    # 随机抽样保留的帧索引，并排序以保持时间流的单向性
    # train_indices 即为我们要用来训练的帧的索引
    train_indices = sorted(random.sample(indices, int(total_frames * keep_ratio)))
    
    # 构建新的、稀疏的训练集 (Irregular Training Set)
    final_train_cameras = [source_train_cam_infos[i] for i in train_indices]
    
    # 构建被丢弃的帧作为测试集 (Held-out Test Set)
    # 这些帧正好位于训练帧的空隙中，用于验证模型的插值能力
    final_test_cameras = [source_train_cam_infos[i] for i in indices if i not in train_indices]
    
    print(f"Original training frames: {total_frames}")
    print(f"Irregular training frames (kept): {len(final_train_cameras)}")
    print(f"Held-out test frames (interpolated): {len(final_test_cameras)}")
    
    # ------------------------------------------------------------------

    # 注意：原本的测试集 cam_infos[train_num:] 在此实验中不再使用，
    # 因为我们的目的是验证对训练序列内部缺失帧的恢复能力。

    # 基于新的训练集计算归一化参数
    nerf_normalization = getNerfppNorm(final_train_cameras)

    # 点云处理 (保持原逻辑)
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print(f"Generating point cloud from nerfies...")

        xyz = np.load(os.path.join(path, "points.npy"))
        xyz = (xyz - scene_center) * scene_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # 构建最终的 SceneInfo
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=final_train_cameras, # 70% 随机帧
                           test_cameras=final_test_cameras,   # 30% 丢弃帧
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromNpy(path, npy_file, split, hold_id, num_images):
    cam_infos = []
    video_paths = sorted(glob(os.path.join(path, 'frames/*')))
    poses_bounds = np.load(os.path.join(path, npy_file))

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]

    n_cameras = poses.shape[0]
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])

    i_test = np.array(hold_id)
    video_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in video_list:
        video_path = video_paths[i]
        c2w = poses[i]
        images_names = sorted(os.listdir(video_path))
        n_frames = num_images

        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        for idx, image_name in enumerate(images_names[:num_images]):
            image_path = os.path.join(video_path, image_name)
            image = Image.open(image_path)
            frame_time = idx / (n_frames - 1)

            FovX = focal2fov(focal, image.size[0])
            FovY = focal2fov(focal, image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
                                        image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], fid=frame_time))

            idx += 1
    return cam_infos


def readPlenopticVideoDataset(path, eval, num_images, hold_id=[0]):
    print("Reading Training Camera")
    train_cam_infos = readCamerasFromNpy(path, 'poses_bounds.npy', split="train", hold_id=hold_id,
                                         num_images=num_images)

    print("Reading Training Camera")
    test_cam_infos = readCamerasFromNpy(
        path, 'poses_bounds.npy', split="test", hold_id=hold_id, num_images=num_images)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, 'points3D.ply')
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "plenopticVideo": readPlenopticVideoDataset,  # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
}
