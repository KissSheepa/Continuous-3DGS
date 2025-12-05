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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim # Original SSIM, will be replaced with MS-SSIM
# from lpipsPyTorch import lpips
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

# Import pytorch_msssim for MS-SSIM
from pytorch_msssim import ssim as ms_ssim_fn # Renamed to avoid conflict with original ssim


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {} # This part seems unused in the evaluation, but kept as is
    per_view_dict_polytopeonly = {} # This part seems unused in the evaluation, but kept as is
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                if not method.startswith("ours"):
                    continue
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ms_ssims = [] # Changed from ssims to ms_ssims
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ms_ssims.append(ms_ssim_fn(renders[idx], gts[idx], data_range=1.0)) # Calculate MS-SSIM, data_range=1.0 for [0, 1] images
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                print("  MS-SSIM : {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5")) # Changed output to MS-SSIM
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS(Alex): {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5")) # Changed output to LPIPS(Alex)
                print("")

                full_dict[scene_dir][method].update({"MS-SSIM": torch.tensor(ms_ssims).mean().item(), # Changed SSIM to MS-SSIM in dict
                                                     "PSNR": torch.tensor(psnrs).mean().item(),
                                                     "LPIPS(Alex)": torch.tensor(lpipss).mean().item()}) # Changed LPIPS key to LPIPS(Alex)
                per_view_dict[scene_dir][method].update(
                    {"MS-SSIM": {name: mssim for mssim, name in zip(torch.tensor(ms_ssims).tolist(), image_names)}, # Changed SSIM to MS-SSIM in dict
                     "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                     "LPIPS(Alex)": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}}) # Changed LPIPS key to LPIPS(Alex)

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device) # Changed net to 'alex' for LPIPS(Alex)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)