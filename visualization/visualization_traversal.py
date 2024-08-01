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

from model.ModelInterface import ModelInterface
from dataset.DataInterface import DataInterface
from utils.vis_utils import reshape_to_image_grid, gpu2np

@hydra.main(version_base="1.1",config_path="../config", config_name="traversal")
def vis_input_output(cfg_dict:DictConfig):
    L.seed_everything(cfg_dict.seed)
    # setup wandb for visualization save 

    exp_name = f"traverse-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    id = wandb.util.generate_id()

    wandb.require("core")
    wandb.init(
        project=cfg_dict.wandb.project,
        entity=cfg_dict.wandb.entity,
        name=exp_name,
        dir=cfg_dict.wandb.dir,
        config=OmegaConf.to_container(cfg_dict, resolve=True),
        resume='allow',
        id = id
    )
    # load ckpt
    model = ModelInterface.load_from_checkpoint(cfg_dict.model.ckpt_path)
    model.eval()
    # load data
    data = DataInterface(**cfg_dict.dataset)
    data.setup(stage='test')
    batch = next(iter(data.test_dataloader())).cuda()
    # latent
    latent = model.encode(batch)
    # latent space variance statistics
    mean = latent.mean(dim=0)
    std = latent.std(dim=0)
    # traverse and visualize each dim
    grid_len = cfg_dict.vis.grid_len
    latent_dim = latent.size(1)
    latent_grid = torch.zeros(grid_len,latent_dim).cuda() + mean
    for i in range(latent_dim):
        grid = torch.linspace(-1.2,1.2,grid_len).cuda()
        output_latent = latent_grid.clone()
        output_latent[:,i] = mean[i] + grid * std[i]
        img = model.decode(output_latent)
        img = reshape_to_image_grid(img, 1)
        wandb.log({f"Traverse Image": wandb.Image(gpu2np(img),caption=f"Z_{i}")},step=i)

    #since w&b already save images locally, just skip saving images.
    wandb.finish()

if __name__ == "__main__":
    vis_input_output()