from dataclasses import dataclass
from typing import Optional
import torch
from pathlib import Path
@dataclass
class BaseModelConfig:
    seed: int = 1
    in_features: int = 1
    out_features: int = 1
    dtype: torch.dtype = torch.float64
@dataclass
class PolyModel1DConfig(BaseModelConfig):
    w0: float = 2.0
    wmin: float = -4.0
    wmax: float = 4.0
    d1: int = 1
    d2: int = 2
    w_init: Optional[torch.Tensor] = None

@dataclass
class PolyModel2DConfig(BaseModelConfig):
    wxm: float = 4.0
    wym: float = 4.0
    d: int = 2
    wx_init: Optional[torch.Tensor] = None
    wy_init: Optional[torch.Tensor] = None

@dataclass
class TrainerConfig:
    dtype: torch.dtype = torch.float64
    nSGD: int = 10**4 # Number of SGD trajectories
    nsamples: int = 10**3
    batch_size: int = 30
    lr: float = 0.01
    momentum: float = 0
    auto: bool = True
    seed: int = 1
    shuffle: bool = True
    output_dir: Path = Path("../../data/")
    save_results: bool = True