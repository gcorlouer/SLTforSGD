import torch
from lib.models.models import PolyModel, PolyModel1DConfig

def test_initialize_weights():
    cfg = PolyModel1DConfig()
    model = PolyModel(cfg)
    model = model._initialize_weights()
    assert model is not None
    assert model.weight is not None
    assert model.weight.shape == (cfg.out_features, cfg.in_features)
    # Initialize with specific weights
    cfg.w_init = torch.tensor([0.3])
    model.update_params(w_init=cfg.w_init)
    assert torch.allclose(model.weight, cfg.w_init)

def test_forward_poly_model_1d():
    cfg = PolyModel1DConfig()
    model = PolyModel(cfg)
    model.reset_parameters()
    input = model.w0
    output = model(input)
    assert output.item() == 0.0, f"Output should be {model.w0} at the local minimum, got {output.item()}"
    input = - model.w0
    output = model(input)
    assert output.item() == 0.0, f"Output should be {model.w0} at the local maximum, got {output.item()}"




