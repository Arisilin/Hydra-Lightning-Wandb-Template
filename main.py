import torch
import numpy 
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import os

from model.ModelInterface import ModelInterface
from dataset.DataInterface import DataInterface

@hydra.main(version_base="1.1",config_path="config", config_name="main")
def train(cfg_dict: DictConfig):
    L.seed_everything(cfg_dict.seed)

    # setup wandb for data loaders and model setup, since PL loggers cant support these cases. 
    # and later we resume the experiment for the trainer.
    exp_name = f"exp{HydraConfig.get().job.id}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" if HydraConfig.get().mode == RunMode.MULTIRUN else f"exp-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    id = wandb.util.generate_id()
    
    wandb.init(project=cfg_dict.wandb.project, 
               entity=cfg_dict.wandb.entity, 
               dir=cfg_dict.wandb.dir, 
               name=exp_name,
               config=OmegaConf.to_container(cfg_dict, resolve=True),
               resume='allow',
               id = id)
    wandb.run.log_code(".", include_fn=lambda path: path.endswith('.py')
                    or path.endswith('.yaml')
                    or path.endswith('.sh')
                    or path.endswith('.txt'))
    #Setup Model Interface
    mi = ModelInterface(cfg_dict)
    #Setup Dataset Loader Interface
    di = DataInterface(**cfg_dict.dataset)
    wandb.finish()
    
    #Setup wandb
    if cfg_dict.wandb.enabled == True:
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            entity=cfg_dict.wandb.entity,
            name=exp_name,
            tags=cfg_dict.wandb.tags,
            dir=cfg_dict.wandb.dir,
            notes=HydraConfig.get().job.override_dirname+cfg_dict.wandb.note if HydraConfig.get().mode == RunMode.MULTIRUN else cfg_dict.wandb.note,
            config=OmegaConf.to_container(cfg_dict, resolve=True),
            id = id,
            resume='allow'
        )

    #Setup Callbacks
    callbacks = []
    ckpt_path = cfg_dict.lightning.multirun_ckpt_save_dir if HydraConfig.get().mode == RunMode.MULTIRUN else cfg_dict.lightning.ckpt_save_dir
    callbacks.append(ModelCheckpoint(dirpath=ckpt_path,verbose=True,every_n_train_steps=30))

    #setup trainer
    trainer = L.Trainer(**cfg_dict.trainer,
                        logger=logger if cfg_dict.wandb.enabled == True else None,
                        log_every_n_steps=20,
                        # default_root_dir=cfg_dict.lightning.multirun_ckpt_save_dir if HydraConfig.get().mode == RunMode.MULTIRUN else cfg_dict.lightning.ckpt_save_dir
                        callbacks=callbacks
                        )

    #fit model.
    trainer.fit(mi, di)

if __name__ == "__main__":
    train()

