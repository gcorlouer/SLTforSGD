import pytest
import torch

from lib.training.trainer import SGDTrainer
from lib.config.model_config import TrainerConfig, PolyModel1DConfig
from lib.models.models import PolyModel

@pytest.fixture
def trainer_config():
    return TrainerConfig(nSGD=10,nsamples=50,batch_size=5,seed=1)

@pytest.fixture
def model_config():
    return PolyModel1DConfig(w0=2.0, wmin=-4.0, wmax=4.0, d1=1, d2=2, w_init=None, seed=1)


class TestTrainer:
    def test_make_dataset(self, trainer_config):
        # Test normal input
        trainer = SGDTrainer(trainer_config)
        dataset = trainer.make_dataset(nfeatures=1)
        nsamples = len(dataset) * dataset.batch_size
        assert nsamples == trainer.config.nsamples, f"Number of samples is {nsamples}, should be {trainer.config.nsamples}"
        assert dataset.batch_size == trainer.config.batch_size, f"Batch size is {dataset.batch_size}, should be {trainer.config.batch_size}"
        x_data, y_data = next(iter(dataset))
        assert x_data.shape == (trainer.config.batch_size, 1), f"x_data shape is {x_data.shape}, should be {(trainer.config.batch_size, 1)}"
        assert y_data.shape == (trainer.config.batch_size, 1), f"y_data shape is {y_data.shape}, should be {(trainer.config.batch_size, 1)}"
        # Test that batches are shuffled
        dataset_iterator = iter(dataset)
        x_data, y_data = next(dataset_iterator)
        x_data_2, y_data_2 = next(dataset_iterator)
        assert not torch.allclose(x_data, x_data_2), f"x_data and x_data_2 are the same"
        assert not torch.allclose(y_data, y_data_2), f"y_data and y_data_2 are the same"
        # Test edge cases
        config = TrainerConfig(nsamples=1, batch_size=1, nSGD=1)
        trainer = SGDTrainer(config)
        dataset = trainer.make_dataset(nfeatures=1)
        assert dataset.batch_size == 1, f"Batch size is {dataset.batch_size}, should be {1}"
        x_data, y_data = next(iter(dataset))
        assert x_data.shape == (1, 1), f"x_data shape is {x_data.shape}, should be {(1, 1)}"
        assert y_data.shape == (1, 1), f"y_data shape is {y_data.shape}, should be {(1, 1)}"

    def test_dataset_reproducibility(self, trainer_config):
        """Test that datasets are reproducible with same seed."""
        trainer1 = SGDTrainer(trainer_config)
        trainer2 = SGDTrainer(trainer_config)
        
        dataset1 = trainer1.make_dataset(nfeatures=1)
        dataset2 = trainer2.make_dataset(nfeatures=1)
        
        # Get first few batches from each dataset
        batches1 = [next(iter(dataset1)) for _ in range(3)]
        batches2 = [next(iter(dataset2)) for _ in range(3)]
        
        # Compare x and y data for each batch
        for (x1, y1), (x2, y2) in zip(batches1, batches2):
            assert torch.allclose(x1, x2), "Dataset x values not reproducible"
            assert torch.allclose(y1, y2), "Dataset y values not reproducible"
    
    def test_train(self, model_config):
        trainer_config = TrainerConfig(nSGD=100, nsamples=200, batch_size=20, lr=0.01, seed=1)
        trainer = SGDTrainer(trainer_config)
        model = PolyModel(model_config)
        dataset = trainer.make_dataset(nfeatures=1)
        running_loss, running_weight = trainer.train(model, dataset)
        # Check if loss decreases
        assert running_loss[-1] < running_loss[0], "Loss should decrease during training"
        
        # Check if weights are changing
        assert not all(w == running_weight[0] for w in running_weight), "Weights should change during training"

    def test_different_batch_sizes(self, model_config):
        """Test training with different batch sizes."""
        batch_sizes = [1, 5, 50]
        for batch_size in batch_sizes:
            config = TrainerConfig(batch_size=batch_size, nsamples=100)
            trainer = SGDTrainer(config)
            model = PolyModel(model_config)
            dataset = trainer.make_dataset(nfeatures=1)
            losses, _ = trainer.train(model, dataset)
            assert len(losses) > 0, f"Training failed with batch_size={batch_size}"

    def test_learning_rate_effect(self, model_config):
        """Test that learning rate affects training."""
        lr_configs = [
            TrainerConfig(lr=0.001, nSGD=100),
            TrainerConfig(lr=0.1, nSGD=100)
        ]
        losses_per_lr = []
        
        for config in lr_configs:
            trainer = SGDTrainer(config)
            model = PolyModel(model_config)
            dataset = trainer.make_dataset(nfeatures=1)
            losses, _ = trainer.train(model, dataset)
            losses_per_lr.append(losses)
        
        # Different learning rates should lead to different training paths
        assert not torch.allclose(
            torch.tensor(losses_per_lr[0]), 
            torch.tensor(losses_per_lr[1])
        ), "Different learning rates should produce different loss curves"

    def test_reproducibility(self, trainer_config, model_config):
        """Test that training is reproducible with same seed."""
        trainer1 = SGDTrainer(trainer_config)
        trainer2 = SGDTrainer(trainer_config)
        
        model1 = PolyModel(model_config)
        model2 = PolyModel(model_config)
        
        dataset1 = trainer1.make_dataset(nfeatures=1)
        dataset2 = trainer2.make_dataset(nfeatures=1)
        
        losses1, weights1 = trainer1.train(model1, dataset1)
        losses2, weights2 = trainer2.train(model2, dataset2)
        
        assert torch.allclose(
            torch.tensor(losses1), 
            torch.tensor(losses2)
        ), "Training should be reproducible with same seed"

