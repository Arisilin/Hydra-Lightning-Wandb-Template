import torch
import numpy 
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from einops import rearrange
import os

def reshape_to_image_grid(x: torch.Tensor, nrow: int) -> torch.Tensor:
    x = rearrange(x, "(b1 b2) c h w -> c (b1 h) (b2 w)",b1=nrow)
    x = rearrange(x,"c h w -> h w c")
    return x

def gpu2np(x: torch.Tensor) -> numpy.ndarray:
    return x.cpu().detach().numpy()

def cpu2np(x: torch.Tensor) -> numpy.ndarray:
    return x.detach().numpy()

