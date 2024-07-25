import os
from dataclasses import dataclass
from torch import nn,optim,utils,Tensor
import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger
from .decoder import Decoder,DecoderCfg
from typing import Optional
from einops import rearrange, repeat, pack, reduce

@dataclass
class VAEConvDecoderCfg(DecoderCfg):
    img_size: int
    map_size: int 
    latent_dim: int
    kernel_sizes: list[int]
    strides: list[int]
    paddings: list[int]
    hidden_dims: list[int]
    in_channel: int

class VAEConvDecoder(Decoder[VAEConvDecoderCfg]):
    def __init__(
        self,
        cfg: VAEConvDecoderCfg,
    ):
        super().__init__(cfg)
        self.img_size = cfg.img_size
        self.map_size = cfg.map_size
        self.latent_dim = cfg.latent_dim
        self.kernel_sizes = cfg.kernel_sizes
        self.strides = cfg.strides
        self.paddings = cfg.paddings
        self.hidden_dims = cfg.hidden_dims
        self.in_channel = cfg.in_channel

        self.fc = nn.Sequential(nn.Linear(self.latent_dim,self.latent_dim),
                                nn.ReLU(),
                                nn.Linear(self.latent_dim,self.latent_dim),
                                nn.ReLU(),
                                nn.Linear(self.latent_dim, self.hidden_dims[-1]*self.map_size*self.map_size),
                                nn.ReLU())
        self.deConvLayers = nn.ModuleList()
        self.outputLayers = nn.ModuleList()
        for i in range(len(self.kernel_sizes)):
            
            self.deConvLayers.append(nn.ConvTranspose2d(
                in_channels= self.hidden_dims[-1-i],
                out_channels=self.in_channel*4 if i == len(self.kernel_sizes)-1 else self.hidden_dims[-2-i],
                kernel_size=self.kernel_sizes[-1-i],
                stride=self.strides[-1-i],
                padding=self.paddings[-1-i]
            ))
            self.deConvLayers.append(nn.BatchNorm2d(self.in_channel*4 if i == len(self.kernel_sizes)-1 else self.hidden_dims[-2-i]))
            self.deConvLayers.append(nn.LeakyReLU()) 
            
        self.outputLayers.append(nn.Conv2d(self.in_channel*4,self.in_channel,3,1,1))
        self.to_img = nn.Sigmoid()
    def forward(self, x):
        x = self.fc(x)
        x = rearrange(x, 'b (c h w) -> b c h w', c=self.hidden_dims[-1], h=self.map_size, w=self.map_size)
        for layer in self.deConvLayers:
            x = layer(x)
        for layer in self.outputLayers:
            x = layer(x)
        x = self.to_img(x)
        x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear')
        return x

