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

@hydra.main(version_base="1.1",config_path="../config", config_name="visualization")
def vis_input_output(cfg_dict:DictConfig):
    L.seed_everything(cfg_dict.seed)
    # setup wandb for visualization save 

    exp_name = f"vis-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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
    batch = next(iter(data.test_dataloader()))[:cfg_dict.vis.visualize_batch].cuda()
    # recon
    recon = model(batch)
    #visualize
    vis_fn = lambda x: reshape_to_image_grid(x, cfg_dict.vis.visualize_nrow)
    origin = wandb.Image(gpu2np(vis_fn(batch)),caption="original")
    recon = wandb.Image(gpu2np(vis_fn(recon)),caption="recon")
    wandb.log({"original & recon": [origin,recon]})

    #since w&b already save images locally, just skip saving images.
    wandb.finish()

if __name__ == "__main__":
    vis_input_output()
    
    