from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn

T = TypeVar("T")

class DecoderCfg(ABC):
    pass

class Decoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, x):
        pass
    