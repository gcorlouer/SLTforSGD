import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import copy
from pathlib import Path
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

@dataclass
class TrainerConfig:
    batch_size: int = 30
    nsamples: int = 1000
    lr: float = 0.01
    momentum: float = 0.9
    shuffle: bool = True
    seed: int = 1

class SGDTrainer:
    """Handles training of models using SGD."""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate trainer configuration."""
        if self.config.batch_size > self.config.nsamples:
            raise ValueError(
                f"Batch size ({self.config.batch_size}) cannot be larger "
                f"than number of samples ({self.config.nsamples})"
            )
        if self.config.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.config.lr}")

    def make_dataset(self, model) -> DataLoader:
        """Create training dataset."""
        torch.manual_seed(self.config.seed)
        x_data = torch.randn((self.config.nsamples, model.out_features, model.in_features))
        y_data = torch.randn((self.config.nsamples, model.out_features, model.in_features))
        dataset = TensorDataset(x_data, y_data)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle)

    def train(self, model, dataset: DataLoader) -> Tuple[list, list]:
        """Train model with error handling and logging."""
        try:
            logger.info(f"Starting training with batch_size={self.config.batch_size}")
            running_loss = []
            running_weight = []
            
            if hasattr(model, 'w_init') and model.w_init is not None:
                w_init_copy = copy.deepcopy(model.w_init.clone())
                running_weight = [w_init_copy.item()]
            
            model.double()
            loss_function = nn.MSELoss()
            
            if self.config.auto:
                optimizer = optim.SGD(
                    model.parameters(), 
                    momentum=self.config.momentum,
                    lr=self.config.lr
                )

            for xb, yb in dataset:
                xb, yb = xb.double(), yb.double()
                
                if self.config.auto:
                    y_pred = model(xb)
                    loss = loss_function(y_pred, yb)
                    running_loss.append(loss.item())
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    self._manual_sgd_step(model, xb, yb)
                
                if hasattr(model, 'weight'):
                    running_weight.append(model.weight.item())
                
            logger.info("Training completed successfully")
            return running_loss, running_weight
            
        except RuntimeError as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def _manual_sgd_step(self, model, xb, yb):
        """Perform manual SGD step."""
        grad = model.gradient()
        predicted = model(xb)
        error = predicted - yb
        loss_grad = 2 * xb * error / xb.size(0)
        update = model.weight - self.config.lr * grad * torch.sum(loss_grad, dim=0)
        model.weight.data = update 