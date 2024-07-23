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
    model = ModelInterface()