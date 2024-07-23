from typing import Optional


from .encoder import Encoder
from .encoder_Conv import VAEConvEncoder, VAEConvEncoderCfg

ENCODERS = {
    "conv": (VAEConvEncoder),
}

def get_encoder(cfg):
    encoder = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    return encoder