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

@hydra.main(version_base="1.1",config_path="config", config_name="main")
def train(cfg_dict: DictConfig):
    print(os.getcwd())
    L.seed_everything(cfg_dict.seed)

    # setup wandb for data loaders and model setup, since PL loggers cant support these cases. 
    # and later we resume the experiment for the trainer.
    exp_name = f"exp-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    id = wandb.util.generate_id()
    wandb.init(project=cfg_dict.wandb.project, entity=cfg_dict.wandb.entity, name=exp_name, config=OmegaConf.to_container(cfg_dict, resolve=True),resume='allow',id = id)
    
    #Setup Model Interface
    mi = ModelInterface(cfg_dict)
    #Setup Dataset Loader Interface
    di = DataInterface(**cfg_dict.dataset)

    wandb.finish()
    callbacks = []
    #Setup wandb
    
    if cfg_dict.wandb.enabled == True:
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            entity=cfg_dict.wandb.entity,
            name=exp_name,
            tags=cfg_dict.wandb.tags,
            notes=cfg_dict.wandb.note,
            config=OmegaConf.to_container(cfg_dict, resolve=True),
            id = id,
            resume='allow'
        )

    trainer = L.Trainer(**cfg_dict.trainer,
                        logger=logger if cfg_dict.wandb.enabled == True else None,
                        log_every_n_steps=20,
                        )
    trainer.fit(mi, di)

if __name__ == "__main__":
    train()

