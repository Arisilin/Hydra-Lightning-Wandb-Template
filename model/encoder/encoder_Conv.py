import os
from dataclasses import dataclass
from torch import nn,optim,utils,Tensor
import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger
from .encoder import Encoder, EncoderCfg
from typing import Optional
from einops import rearrange, repeat, pack, reduce
from utils.model_utils import calculate_convmap_size

@dataclass
class VAEConvEncoderCfg(EncoderCfg):
    img_size: int
    # map_size: int 
    latent_dim: int
    kernel_sizes: list[int]
    strides: list[int]
    paddings: list[int]
    hidden_dims: list[int]
    in_channel: int

class VAEConvEncoder(Encoder[VAEConvEncoderCfg]):
    def __init__(
        self,
        cfg: VAEConvEncoderCfg,
    ):
        super().__init__(cfg)
        self.img_size = cfg.img_size
        self.latent_dim = cfg.latent_dim
        self.kernel_sizes = cfg.kernel_sizes
        self.strides = cfg.strides
        self.paddings = cfg.paddings
        self.hidden_dims = cfg.hidden_dims
        self.in_channel = cfg.in_channel
        self.ConvLayers = nn.ModuleList()
        for i in range(len(self.kernel_sizes)):
            self.ConvLayers.append(nn.Conv2d(
                in_channels=self.in_channel if i == 0 else self.hidden_dims[i-1],
                out_channels=self.hidden_dims[i],
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                padding=self.paddings[i]
            ))
            self.ConvLayers.append(nn.BatchNorm2d(self.hidden_dims[i]))
            self.ConvLayers.append(nn.LeakyReLU())

        self.map_size = calculate_convmap_size(self.img_size, self.ConvLayers)
        if (ms:=self.map_size) < 2:
            raise ValueError("Feature map too small.")
        else:
            self.fc_mu = nn.Sequential(
                nn.Linear(self.hidden_dims[-1]*ms*ms, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim)
            )
            self.fc_var = nn.Sequential(
                nn.Linear(self.hidden_dims[-1]*ms*ms, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim)
            )

    def forward(self, x):
        #downscale for too large images
        x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear')
        for layer in self.ConvLayers:
            x = layer(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    

        
