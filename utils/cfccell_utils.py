try:
    import torch
except:
    raise ImportWarning(
        "It seems like the PyTorch package is not installed\n"
        "Installation instructions: https://pytorch.org/get-started/locally/\n",
    )
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union
from torch_points3d.applications.kpconv import KPConv


class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class PointwiseConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, activation=nn.ReLU):
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(nn.Conv1d(current_channels, out_channels, kernel_size=1)) 
            layers.append(activation()) 
            current_channels = out_channels 

        self.pointwise_conv_layers = nn.Sequential(*layers) 

    def forward(self, x):

        x = x.unsqueeze(0).permute(0, 2, 1) 

        x = self.pointwise_conv_layers(x)

        x = x.squeeze(0).permute(1, 0) 
        return x


class CfCCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        mode="default",
        backbone_activation="lecun_tanh",
        backbone_units=128,
        backbone_layers=1,
        backbone_dropout=0.0,
        sparsity_mask=None,
    ):

        super(CfCCell, self).__init__()

        self.time_scale = nn.Parameter(torch.ones(1))
        self.input_size = input_size
        self.hidden_size = hidden_size
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                f"Unknown mode '{mode}', valid options are {str(allowed_modes)}"
            )
        self.sparsity_mask = (
            None
            if sparsity_mask is None
            else torch.nn.Parameter(
                data=torch.from_numpy(np.abs(sparsity_mask.T).astype(np.float32)),
                requires_grad=False,
            )
        )

        self.mode = mode

        if backbone_activation == "silu":
            backbone_activation = nn.SiLU
        elif backbone_activation == "relu":
            backbone_activation = nn.ReLU
        elif backbone_activation == "tanh":
            backbone_activation = nn.Tanh
        elif backbone_activation == "gelu":
            backbone_activation = nn.GELU
        elif backbone_activation == "lecun_tanh":
            backbone_activation = LeCun
        else:
            raise ValueError(f"Unknown activation {backbone_activation}")

        self.backbone = None
        self.backbone_layers = backbone_layers
        if backbone_layers > 0:
            self.backbone_input = nn.Sequential(
                nn.Linear(input_size, backbone_units),
                backbone_activation(),
                nn.Linear(backbone_units, backbone_units),
                backbone_activation(),
            )
            self.backbone_hx = nn.Sequential(
                nn.Linear(160, backbone_units),
                backbone_activation(),
                nn.Linear(backbone_units, backbone_units),
                backbone_activation(),
            )

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        cat_shape = int(
            self.hidden_size + input_size if backbone_layers == 0 else backbone_units
        )
        self.ff1 = PointwiseConvBlock1D(
            in_channels=cat_shape,
            out_channels=hidden_size,
            num_layers=2,
            activation=nn.ReLU
        )
        self.ff2 = PointwiseConvBlock1D(
            in_channels=cat_shape,
            out_channels=hidden_size,
            num_layers=2,
            activation=nn.ReLU
        )
        self.time_a = nn.Linear(cat_shape * 2, hidden_size)
        self.time_b = nn.Linear(cat_shape * 2, hidden_size)
        # self.time_a = nn.Linear(hidden_size, hidden_size)
        # self.time_b = nn.Linear(hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)


    def forward(self, input, hx, ts):
        # x = torch.cat([input, hx], 1)
        if self.backbone_layers > 0:
            input = self.backbone_input(input)
            hx = self.backbone_hx(hx)
            # x = self.backbone(x)
            x = torch.cat([input, hx], 1)

        ff1 = self.ff1(hx)
        ff2 = self.ff2(input)
        ff1 = self.tanh(ff1)
        ff2 = self.tanh(ff2)
        t_a = self.time_a(x)
        t_b = self.time_b(x)
        # t_interp = self.sigmoid(t_a * ts + t_b)
        t_interp = self.sigmoid(self.time_scale * t_a * ts + t_b)
        # t_interp = self.sigmoid(t_a + t_b)
        new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden, new_hidden