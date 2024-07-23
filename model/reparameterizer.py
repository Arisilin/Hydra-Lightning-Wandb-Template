from torch import nn
import torch

class reparameterizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
def get_reparameterizer(cfg=None):
    return reparameterizer(cfg)