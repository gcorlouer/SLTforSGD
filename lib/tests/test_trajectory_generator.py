import pytest

from lib.training.trajectory_generator import ParameterSweeper
from lib.config.model_config import PolyModel1DConfig, TrainerConfig

@pytest.fixture
def parameter_sweeper():
    return ParameterSweeper(PolyModel1DConfig(), TrainerConfig())


class TestGenerator:
    def test_generate_initial_weights(self):
        generator = TrajectoryGenerator(PolyModel1DConfig(), TrainerConfig())
        initial_weights = generator._generate_initial_weights()
        assert initial_weights.shape == (1, 1, 10)

    def test_generate(self):
        # Test normal input
        generator = ParameterSweeper(PolyModel1DConfig(), TrainerConfig())
        trajectories = generator.generate()
        assert trajectories.shape == (10, 1, 10)
        # Test edge cases   

class TestParameterSweeper:
    def test_regular_fraction(self):
        pass 

    def test_compute_escape_rate(self):
        pass

    def test_process_trajectories(self):
        pass    

    def _save_results(self):
        pass

    def test_parameter_sweep(self):
        pass

