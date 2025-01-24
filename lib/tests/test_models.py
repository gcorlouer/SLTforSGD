import torch
import pytest
from lib.models.models import PolyModel, PolyModel1DConfig, PolyModel2DConfig, PolyModel2D
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

def test_compute_terms(model_config):
    model = PolyModel(model_config)
    w = torch.tensor([model.w0])
    w1, w2 = model._compute_terms(w)
    assert w2.item() == 0, f"w2 should be 0, is {w2.item()}"
    w = torch.tensor([-model.w0])
    w1, w2 = model._compute_terms(w)
    assert w1.item() == 0, f"w1 should be 0, is {w1.item()}"

def test_forward_poly_model_1d(model_config):
    model = PolyModel(model_config)
    input = torch.tensor([0])
    output = model(input)
    assert output.item() == 0.0, f"Output should be 0 got {output.item()}"
    input = torch.tensor([1])
    output = model(input)
    w1, w2 = model._compute_terms(model.weight.detach())
    assert output.item() == w1 * w2 * input.item(), f"Output should be {w1 * w2 * input.item()}, got {output.item()}"

def test_initialize_weights_2d():
    model_config = PolyModel2DConfig()
    model = PolyModel2D(model_config)
    weights = model.weights[0]
    print(weights)
    assert model is not None, f"Model should be initialized, is {model}"
    assert weights is not None, f"Model should have weights, is {model.weights}"
    assert weights.shape == (model_config.out_features, model_config.in_features), f"Model should have weights of shape {model_config.out_features, model_config.in_features}, is {weights.shape}"
    wtest = model.weights[0]
    model._initialize_weights()
    winit = model.weights[0]
    assert torch.allclose(winit, wtest), f"Model should have weights {wtest}, is {winit}"
    winit = torch.ones(model_config.out_features, model_config.in_features)
    model_config = PolyModel2DConfig(wx_init=winit, wy_init=winit)
    model = PolyModel2D(model_config)
    w1, w2 = model._initialize_weights()
    assert torch.allclose(w1, winit), f"Model should have weights {winit}, is {w1}"
    assert torch.allclose(w2, winit), f"Model should have weights {winit}, is {w2}"

def test_forward_poly_model_2d():
    model_config = PolyModel2DConfig()
    model = PolyModel2D(model_config)
    input = torch.tensor([0])
    output = model(input)
    assert output.item() == 0.0, f"Output should be 0, got {output.item()}"
    wxinit = torch.ones(model_config.out_features, model_config.in_features)
    wyinit = torch.zeros(model_config.out_features, model_config.in_features)
    model_config = PolyModel2DConfig(wx_init=wxinit, wy_init=wyinit)
    model = PolyModel2D(model_config)
    input = torch.tensor([1])
    output = model(input)
    assert output.item() == 1, f"Output should be 1, got {output.item()}"

