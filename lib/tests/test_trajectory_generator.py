import pytest
import torch
import numpy as np

from lib.training.trajectory_generator import TrajectoryGenerator
from lib.training.trajectory_generator import ParameterSweeper
from lib.config.model_config import PolyModel1DConfig, TrainerConfig
from lib.models.models import PolyModel

@pytest.fixture
def trajectory_config():
    model_config = PolyModel1DConfig()
    trainer_config = TrainerConfig(nSGD=3, nsamples=10, batch_size=2, momentum=0.1)
    return model_config, trainer_config

class TestGenerator:
    def test_generate_initial_weights(self, trajectory_config):
        model_config, trainer_config = trajectory_config
        generator = TrajectoryGenerator(model_config, trainer_config)
        initial_weights = generator._generate_initial_weights()
        assert initial_weights.shape == (1, 1, trainer_config.nSGD), f"Initial weights shape is {initial_weights.shape}, should be (1, 1, {trainer_config.nSGD})"
        assert torch.unique(initial_weights).size(0) == initial_weights.numel(), "Initial weights are not all different"

    def test_generate(self, trajectory_config):
        # Test normal input
        model_config, trainer_config = trajectory_config
        model = PolyModel(model_config)
        model.to(model_config.dtype)
        generator = TrajectoryGenerator(model_config, trainer_config)
        df = generator.generate(model)
        assert len(df["trajectory"]) == trainer_config.nSGD, f"Number of trajectories is {len(df['trajectory'])}, should be {trainer_config.nSGD}"
        assert len(set(df["w_init"])) == len(df["w_init"]), "Initial weights are not all different"
        

class TestParameterSweeper:
    
    def test_regular_fraction(self, trajectory_config):
        model_config, trainer_config = trajectory_config
        model = PolyModel(model_config)
        model.to(model_config.dtype)
        generator = TrajectoryGenerator(model_config, trainer_config)
        df = generator.generate(model)
        parameter_sweeper = ParameterSweeper(model_config, trainer_config)
        trajectories = parameter_sweeper._process_trajectories(df)
        fraction = parameter_sweeper._regular_fraction(trajectories, model_config)
        assert np.all(fraction <= 1), "Fraction is greater than 1"
        assert np.all(fraction >= 0), "Fraction is less than 0"

    def test_compute_escape_rate(self, trajectory_config):
        model_config, trainer_config = trajectory_config
        model = PolyModel(model_config)
        model.to(model_config.dtype)
        generator = TrajectoryGenerator(model_config, trainer_config)
        df = generator.generate(model)
        parameter_sweeper = ParameterSweeper(model_config, trainer_config)
        trajectories = parameter_sweeper._process_trajectories(df)
        fraction = parameter_sweeper._regular_fraction(trajectories, model_config)
        escape_rate, error = parameter_sweeper._compute_escape_rate(fraction)
        assert np.all(escape_rate <= 1), "Escape rate is greater than 1"
        assert np.all(escape_rate >= 0), "Escape rate is less than 0"

    def test_parameter_sweep(self, trajectory_config):
        w0_range = np.linspace(-1, 1, 2)
        batch_range = np.linspace(1, 10, 2)
        lr_range = np.linspace(0.01, 0.1, 2)
        model_config, trainer_config = trajectory_config
        model = PolyModel(model_config)
        model.to(model_config.dtype)
        parameter_sweeper = ParameterSweeper(model_config, trainer_config)
        df = parameter_sweeper.parameter_sweep(w0_range, batch_range, lr_range, model)
        print(df)
