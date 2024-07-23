from typing import Optional


from .decoder import Decoder
from .decoder_Conv import VAEConvDecoder, VAEConvDecoderCfg

ENCODERS = {
    "conv": (VAEConvDecoder),
}

def get_decoder(cfg):
    decoder = ENCODERS[cfg.name]
    decoder = decoder(cfg)
    return decoder