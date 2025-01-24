"""Unit tests for polynomial models.

This module contains test cases for the 1D and 2D polynomial models used in SGD analysis.
Tests cover model initialization, forward passes, parameter updates, and special properties.
"""

import torch
import pytest
from lib.models.models import PolyModel, PolyModel1DConfig, PolyModel2DConfig, PolyModel2D
from pathlib import Path
import sys

@pytest.fixture
def model_config():
    """Fixture providing default 1D polynomial model configuration."""
    return PolyModel1DConfig()

class TestPolyModel1D:
    """Test suite for 1D polynomial model."""

    def test_initialize_weights(self, model_config):
        """Test weight initialization and shape validation.
        
        Tests:
            - Model instantiation
            - Weight initialization
            - Weight shape validation
            - Weight reinitialization consistency
        """
        model = PolyModel(model_config)
        assert model is not None, f"Model should be initialized, is {model}"
        assert model.weight is not None, f"Model should have weights, is {model.weight}"
        assert model.weight.shape == (model_config.out_features, model_config.in_features), f"Model should have weights of shape {model_config.out_features, model_config.in_features}, is {model.weight.shape}"
        # Test initialization
        wtest = model.weight.detach()
        model._initialize_weights()
        winit = model.weight.detach()
        assert torch.allclose(winit, wtest), f"Model should have weights {wtest}, is {winit}"

    def test_compute_terms(self, model_config):
        """Test polynomial term computation at critical points.
        
        Tests computation of w1 and w2 terms at w = ±w0, where one term
        should be zero.
        """
        model = PolyModel(model_config)
        
        # Test at w = w0
        w = torch.tensor([model.w0])
        w1, w2 = model._compute_terms(w)
        assert w2.item() == 0, f"w2 should be 0 at w0, is {w2.item()}"
        
        # Test at w = -w0
        w = torch.tensor([-model.w0])
        w1, w2 = model._compute_terms(w)
        assert w1.item() == 0, f"w1 should be 0 at -w0, is {w1.item()}"

    def test_forward_poly_model_1d(self, model_config):
        """Test forward pass of 1D polynomial model.
        
        Tests:
            - Zero input gives zero output
            - Output matches manual computation
        """
        model = PolyModel(model_config)
        
        # Test zero input
        input = torch.tensor([0])
        output = model(input)
        assert output.item() == 0.0, f"Output should be 0 got {output.item()}"
        input = torch.tensor([1])
        output = model(input)
        w1, w2 = model._compute_terms(model.weight.detach())
        assert output.item() == w1 * w2 * input.item(), f"Output should be {w1 * w2 * input.item()}, got {output.item()}"
    
    def test_parameter_update(self, model_config):
        """Test parameter update functionality.
        
        Verifies that model parameters can be updated correctly through
        the update_params method.
        """
        model = PolyModel(model_config)
        weight = torch.nn.Parameter(torch.tensor([1.0]))
        model.update_params(weight=weight)
        assert model.weight.item() == 1, f"w0 should be 1, got {model.weight.item()}"
    
    def test_barrier_point(self, model_config):
        """Test barrier point computation.
        
        Verifies that the barrier point between phases is computed correctly.
        """
        model = PolyModel(model_config)
        expected = (model.w0 * model.d1 - model.w0 * model.d2) / (model.d1 + model.d2)
        assert model.barrier_point == pytest.approx(expected)

@pytest.fixture
def model_config2d():
    """Fixture providing default 2D polynomial model configuration."""
    return PolyModel2DConfig()

class TestPolyModel2D:
    """Test suite for 2D polynomial model."""

    def test_initialize_weights_2d(self, model_config2d):
        """Test weight initialization for 2D model.
        
        Tests:
            - Model instantiation
            - Weight initialization and shapes
            - Custom weight initialization
        """
        model = PolyModel2D(model_config2d)
        weights = model.weights[0]
        print(weights)
        assert model is not None, f"Model should be initialized, is {model}"
        assert weights is not None, f"Model should have weights, is {model.weights}"
        assert weights.shape == (model_config2d.out_features, model_config2d.in_features), f"Model should have weights of shape {model_config2d.out_features, model_config2d.in_features}, is {weights.shape}"
        wtest = model.weights[0]
        model._initialize_weights()
        winit = model.weights[0]
        assert torch.allclose(winit, wtest), f"Model should have weights {wtest}, is {winit}"
        winit = torch.ones(model_config2d.out_features, model_config2d.in_features)
        model_config2d = PolyModel2DConfig(wx_init=winit, wy_init=winit)
        model = PolyModel2D(model_config2d)
        w1, w2 = model._initialize_weights()
        assert torch.allclose(w1, winit), f"Model should have weights {winit}, is {w1}"
        assert torch.allclose(w2, winit), f"Model should have weights {winit}, is {w2}"


    def test_forward_poly_model_2d(self, model_config2d):
        """Test forward pass of 2D polynomial model.
        
        Tests:
            - Zero input gives zero output
            - Output with specific weight initialization
        """
        model = PolyModel2D(model_config2d)
        
        # Test zero input
        input = torch.tensor([0])
        output = model(input)
        assert output.item() == 0.0, f"Output should be 0, got {output.item()}"
        wxinit = torch.ones(model_config2d.out_features, model_config2d.in_features)
        wyinit = torch.zeros(model_config2d.out_features, model_config2d.in_features)
        model_config2d = PolyModel2DConfig(wx_init=wxinit, wy_init=wyinit)
        model = PolyModel2D(model_config2d)
        input = torch.tensor([1])
        output = model(input)
        assert output.item() == 1, f"Output should be 1, got {output.item()}"
        
    def test_parameter_update(self, model_config2d):
        """Test parameter update functionality for 2D model.
        
        Verifies that model parameters can be updated correctly through
        the update_params method.
        """
        model = PolyModel2D(model_config2d)
        wx = torch.nn.Parameter(torch.tensor([1.0]))
        wy = torch.nn.Parameter(torch.tensor([1.0]))
        weights = (wx, wy)
        model.update_params(weights=weights)
        w1 = model.weights[0]
        w2 = model.weights[1]
        assert w1.item() == wx.item(), f"w1 should be {wx.item()}, got {w1.item()}"
        assert w2.item() == wy.item(), f"w2 should be {wy.item()}, got {w2.item()}"
