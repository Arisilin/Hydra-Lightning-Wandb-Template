from torch import nn
import torch

class MSELoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x:torch.Tensor, x_hat, *args, **kwargs):
        reconstruction_loss = torch.nn.functional.mse_loss(x_hat, x)
        return reconstruction_loss

class VAEKLLoss(nn.Module):
    def __init__(self,cfg):
        super().__init__()
    
    def forward(self, mean, log_var, *args,**kwargs):
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/mean.size(0)
        return kl_loss
    
class BVAEKLLoss(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.beta = cfg.beta
    
    def forward(self, mean, log_var,*args, **kwargs):
        kl_loss = -0.5 * self.beta * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / mean.size(0)
        return kl_loss
    
LossDict = {
    "MSELoss": MSELoss,
    "VAEKLLoss": VAEKLLoss,
    "BVAEKLLoss": BVAEKLLoss
}

def get_losses(cfg):
    losses = []
    for loss in cfg.loss_type:
        losses.append(LossDict[loss](cfg))
    return losses