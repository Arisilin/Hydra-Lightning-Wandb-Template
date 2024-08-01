import os
from torch import nn,optim,utils,Tensor
from dataclasses import dataclass
import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger
from typing import Optional
from einops import rearrange, repeat, pack, reduce

from model.encoder.encoder_Conv import VAEConvEncoder
from model.decoder.decoder_Conv import VAEConvDecoder
from config.config import Cfg
from model.encoder import get_encoder
from model.decoder import get_decoder
from loss.losses import get_losses
from model.reparameterizer import get_reparameterizer


@dataclass
class ModelInterfaceCfg:
    model: Cfg
    losses: list[Cfg]
    dataset: Cfg = None
    optimizer: Cfg = None
    

@dataclass
class ModelCfg:
    encoder: Cfg
    decoder: Cfg

class ModelInterface(L.LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    reparameterizer: nn.Module
    decoder: nn.Module
    losses: nn.ModuleList
    def __init__(
            self,
            cfg: ModelInterfaceCfg
     ):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.encoder = get_encoder(cfg.model.encoder)
        cfg.model.decoder.map_size = self.encoder.map_size
        self.decoder = get_decoder(cfg.model.decoder)
        self.reparameterizer = get_reparameterizer()
        self.losses = nn.ModuleList(get_losses(cfg.losses))
        self.optimizer_cfg = cfg.optimizer

        # self.log("losses", ' '.join([loss.__class__.__name__ for loss in self.losses]))
        # self.log("encoder", self.encoder.__class__.__name__)
        # self.log("decoder", self.decoder.__class__.__name__)
        # self.log("reparameterizer", self.reparameterizer.__class__.__name__)

    def training_step(self, batch, batch_idx):
        x = batch
        mean, log_var = self.encoder(x)
        z = self.reparameterizer(mean, log_var)
        x_hat = self.decoder(z)
        loss = 0
        losses = []
        for loss_fn in self.losses:
            losses.append(loss_fn(x=x, x_hat=x_hat, mean=mean, log_var=log_var))
            loss += losses[-1]
        self.log("train_loss", loss, prog_bar=True)
        for loss_fn, loss_val in zip(self.losses, losses):
            self.log(f"{loss_fn.__class__.__name__}", loss_val, prog_bar=True)
        if batch_idx % 50 == 0:
            self.logger.log_image(images=[x[0],x_hat[0]], key = "img & recon")
        return loss
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterizer(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat
    
    def encode(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterizer(mean, log_var)
        return z

    def decode(self, z):
        return self.decoder(z)
        
    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        return optimizer