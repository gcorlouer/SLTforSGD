import torch
import pytest
from lib.models.models import PolyModel, PolyModel1DConfig
from pathlib import Path
import sys

@pytest.fixture
def model_config():
    return PolyModel1DConfig()

def test_initialize_weights(model_config):
    model = PolyModel(model_config)
    assert model is not None, f"Model should be initialized, is {model}"
    assert model.weight is not None, f"Model should have weights, is {model.weight}"
    assert model.weight.shape == (model_config.out_features, model_config.in_features), f"Model should have weights of shape {model_config.out_features, model_config.in_features}, is {model.weight.shape}"
    # Test initialization
    wtest = model.weight.detach()
    model._initialize_weights()
    winit = model.weight.detach()
    assert torch.allclose(winit, wtest), f"Model should have weights {wtest}, is {winit}"

def test_forward_poly_model_1d(model_config):
    model = PolyModel(model_config)
    input = torch.tensor([model.w0])
    output = model(input)
    assert output.item() == 0.0, f"Output should be {model.w0} at the local minimum, got {output.item()}"
    input = - model.w0
    output = model(input)
    assert output.item() == 0.0, f"Output should be {model.w0} at the local maximum, got {output.item()}"