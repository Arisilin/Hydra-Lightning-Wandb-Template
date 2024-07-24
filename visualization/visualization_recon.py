import torch
import numpy 
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import os

from model.ModelInterface import ModelInterface
from dataset.DataInterface import DataInterface

@hydra.main(version_base="1.1",config_path="config", config_name="visualization")
def vis_input_output(cfg_dict:DictConfig):
    L.seed_everything(cfg_dict.seed)
    # setup wandb for visualization save 
    exp_name = f"vis-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    id = wandb.util.generate_id()
    wandb.init(
        project=cfg_dict.wandb.project,
        entity=cfg_dict.wandb.entity,
        name=exp_name,
        config=OmegaConf.to_container(cfg_dict, resolve=True),
        resume='allow',
        id = id
    )

    # load ckpt
    model = ModelInterface.load_from_checkpoint(cfg_dict.model.ckpt_path)
    model.eval()
    # load data
    data = DataInterface(**cfg_dict.dataset)
    