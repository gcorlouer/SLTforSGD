import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from lib.config.model_config import PolyModel1DConfig, PolyModel2DConfig, BaseModelConfig
# TODO: Test models classes and methods
# TODO: Fix weights tensor vs float

class TrivialModel(nn.Module):
    """Trivial, single parameter model."""
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn([1]))

    def forward(self, input: Tensor):
        return input * 0.0


class LinearModel(nn.Module):
    """Simple linear model."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.linear.weight.data.normal_()

    def forward(self, x):
        return self.linear(x)

class BasePolyModel(nn.Module):
    """Base class for polynomial models."""
    
    def __init__(self, cfg: BaseModelConfig):
        super().__init__()
        self.seed = cfg.seed
        self.in_features = cfg.in_features
        self.out_features = cfg.out_features
        torch.manual_seed(self.seed)

    def update_params(self, **kwargs):
        """Update model parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

    def _validate_input(self, input: Tensor):
        """Validate input tensor dimensions."""
        if input.dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got {input.dim()}D")

class PolyModel(BasePolyModel):
    """1D Polynomial model with configurable degrees."""
    
    def __init__(self, cfg: PolyModel1DConfig) -> None:
        super().__init__(cfg)
        self.w0 = cfg.w0
        self.d1 = cfg.d1
        self.d2 = cfg.d2
        self.wmin = cfg.wmin
        self.wmax = cfg.wmax
        self.w_init = cfg.w_init
        self.weight = self._initialize_weights()

    def _initialize_weights(self) -> nn.Parameter:
        """Initialize model weights."""
        weight = nn.Parameter(
            torch.empty((self.out_features, self.in_features))
        )
        if self.w_init is not None:
            if self.w_init.shape != weight.shape:
                raise ValueError(
                    f"Init weight shape {self.w_init.shape} doesn't match "
                    f"expected shape {weight.shape}"
                )
            weight.data = self.w_init
        else:
            nn.init.uniform_(weight, self.wmin, self.wmax)
        return weight

    def _compute_terms(self, w: Tensor) -> tuple[Tensor, Tensor]:
        """Compute polynomial terms."""
        assert self.d1 >= 0 and self.d2 >= 0, "Degrees must be non-negative"
        w1 = (w + self.w0) ** self.d1
        w2 = (w - self.w0) ** self.d2
        return w1, w2

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass with input validation."""
        self._validate_input(input)
        w1, w2 = self._compute_terms(self.weight)
        return input * w1 * w2

    def gradient(self) -> Tensor:
        """Compute analytic gradient."""
        w1, w2 = self._compute_terms(self.weight)
        dw1 = self.d1 * (self.weight + self.w0) ** (self.d1 - 1)
        dw2 = self.d2 * (self.weight - self.w0) ** (self.d2 - 1)
        return dw1 * w2 + w1 * dw2

    @property
    def barrier_point(self) -> float:
        """Compute the barrier point between singular regions."""
        assert self.w0 != 0, "w0 should not be 0"
        return (self.w0 * self.d1 - self.w0 * self.d2) / (self.d1 + self.d2)

class PolyModel2D(BasePolyModel):
    """2D Polynomial model."""
    
    def __init__(
        self,
        cfg: PolyModel2DConfig,
    ) -> None:
        super().__init__(cfg)
        self.d = cfg.d
        self.wxm = cfg.wxm
        self.wym = cfg.wym
        self.wx_init = cfg.wx_init
        self.wy_init = cfg.wy_init
        self.weights = self._initialize_weights()

    def _initialize_weights(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Initialize both weight parameters."""
        shape = (self.out_features, self.in_features)
        w1 = nn.Parameter(torch.empty(shape))
        w2 = nn.Parameter(torch.empty(shape))
        
        if self.wx_init is not None and self.wy_init is not None:
            if self.wx_init.shape != shape or self.wy_init.shape != shape:
                raise ValueError("Init weights shape mismatch")
            w1.data = self.wx_init
            w2.data = self.wy_init
        else:
            nn.init.uniform_(w1, -self.wxm, self.wxm)
            nn.init.uniform_(w2, -self.wym, self.wym)
        return w1, w2

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass with input validation."""
        self._validate_input(input)
        w1, w2 = self.weights
        Q = (w1**2 + w2**2) ** 2 * w1**self.d
        return input * Q

    @property
    def weight(self) -> Tensor:
        """Stack weights for compatibility with trainer."""
        return torch.stack(self.weights)