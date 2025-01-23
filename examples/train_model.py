import sys
from pathlib import Path

# Add the code directory to Python path
code_dir = Path(__file__).parent.parent
sys.path.append(str(code_dir))

from lib.models.models import PolyModel
from lib.training.trainer import SGDTrainer 
from lib.utils.visualization import plot_escape_rate


def main():
    # Create model
    model = PolyModel(w0=2.0, d1=1, d2=2)
    
    # Setup trainer
    trainer = SGDTrainer(
        batch_size=30,
        lr=0.01,
        nSGD=1000
    )
    
    # Create dataset and train
    dataset = trainer.make_dataset(model)
    losses, weights = trainer.train(model, dataset)
    
    # Analyze results
    escape_rate, error = plot_escape_rate(weights, trainer)
    print(f"Escape rate: {escape_rate:.2e} ± {error:.2e}")

if __name__ == "__main__":
    main() 