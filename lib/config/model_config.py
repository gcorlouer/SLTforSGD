from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class BaseModelConfig:
    seed: int = 1
    in_features: int = 1
    out_features: int = 1

@dataclass
class PolyModel1DConfig(BaseModelConfig):
    w0: torch.Tensor = torch.tensor([2.0])
    wmin: torch.Tensor = torch.tensor([-4.0])
    wmax: torch.Tensor = torch.tensor([4.0])
    d1: int = 1
    d2: int = 2
    w_init: Optional[torch.Tensor] = None

@dataclass
class PolyModel2DConfig(BaseModelConfig):
    wxm: torch.Tensor = torch.tensor([4.0])
    wym: torch.Tensor = torch.tensor([4.0])
    d: int = 2
    wx_init: Optional[torch.Tensor] = None
    wy_init: Optional[torch.Tensor] = None

@dataclass
class TrainerConfig:
    nSGD: int = 10**4
    nsamples: int = 10**3
    batch_size: int = 30
    lr: float = 0.01
    momentum: float = 0
    auto: bool = True
    seed: int = 1
    shuffle: bool = True