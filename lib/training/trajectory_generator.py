from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools

from typing import Dict, List, Optional
from lib.config.model_config import PolyModel1DConfig, TrainerConfig
from lib.training.trainer import SGDTrainer
from lib.models.models import PolyModel


class TrajectoryData:
    """Container for trajectory data."""
    def __init__(self):
        self.data: Dict[str, List] = {
            "w_init": [],
            "trajectory": [],
            "loss": []
        }
    
    def add_trajectory(self, w_init: float, trajectory: List[float], loss: List[float]):
        """Add a single trajectory."""
        self.data["w_init"].append(w_init)
        self.data["trajectory"].append(trajectory)
        self.data["loss"].append(loss)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert data to DataFrame."""
        return pd.DataFrame(self.data)

class TrajectoryGenerator:
    """Generates and manages training trajectories."""
    
    def __init__(self, model_config: PolyModel1DConfig, trainer_config: TrainerConfig):
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.trainer = SGDTrainer(trainer_config)
    
    def _generate_initial_weights(self) -> torch.Tensor:
        """Generate initial weights for trajectories."""
        np.random.seed(self.model_config.seed)
        return torch.tensor(
            np.random.uniform(
                self.model_config.wmin,
                self.model_config.wmax,
                size=(self.model_config.out_features, self.model_config.in_features, self.trainer_config.nSGD)
            ),
        )
    
    def _get_filename(self) -> Path:
        """Generate filename for results."""
        model_params = f"w0_{self.model_config.w0}_d1_{self.model_config.d1}_d2_{self.model_config.d2}_seed_{self.model_config.seed}"
        sgd_params = (f"lr_{self.trainer_config.lr}_b_{self.trainer_config.batch_size}"
                     f"_seed_{self.trainer_config.seed}_N_{self.trainer_config.nSGD}"
                     f"_m_{self.trainer_config.nsamples}")
        fname = f"sgd_traj_{sgd_params}_{model_params}.csv"
        return self.trainer_config.output_dir.joinpath(fname)

    def generate(self, model: PolyModel) -> pd.DataFrame:
        """Generate trajectories for the model."""
        dataset = self.trainer.make_dataset(nfeatures=1)
        w_inits = self._generate_initial_weights()
        trajectory_data = TrajectoryData()
        
        for i in range(self.trainer_config.nSGD):
            if (i + 1) % 1000 == 0:
                print(f"Generating trajectory {i+1}/{self.trainer_config.nSGD}")
                
            # Initialize model with new weights
            wi = w_inits[:, :, i]
            model.update_params(w_init=wi)
            model.update_params(weight=torch.nn.Parameter(model.w_init))

            # Train and collect results
            running_loss, running_weight = self.trainer.train(model, dataset)
            trajectory_data.add_trajectory(
                w_init=wi.item(),
                trajectory=running_weight,
                loss=running_loss
            )
        
        df = trajectory_data.to_dataframe()
        
        if self.trainer_config.save_results:
            self._save_results(df, model)
            
        return df
    
    def _save_results(self, df: pd.DataFrame, model: PolyModel):
        """Save results to file."""
        output_path = self._get_filename()
        if not output_path.parent.exists():
            raise FileNotFoundError(f"Output directory '{output_path.parent}' does not exist")
        df.to_csv(output_path)

class ParameterSweeper:
    """Sweeps over parameters and generates trajectories."""
    def __init__(self, model_config: PolyModel1DConfig, trainer_config: TrainerConfig):
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.trajectory_generator = TrajectoryGenerator(model_config, trainer_config)

    def _setup_experiment_tracking(self, w0_range, batch_range, lr_range) -> tuple[int, dict]:
        """Initialize experiment tracking data structures."""
        total_experiments = len(w0_range) * len(batch_range) * len(lr_range)
        results_dict = {
            "escape_rate": [], "lr": [], "B": [], "w0": [],
            "fraction": [], "error": []
        }
        return total_experiments, results_dict

    def _process_trajectories(self, df) -> np.ndarray:
        """Process and clean trajectory data."""
        trajectories = np.asarray(df["trajectory"].to_list())
        return trajectories[~np.isnan(trajectories).any(axis=1)]

    def _regular_fraction(self, trajectories: np.array, model_config: PolyModel1DConfig) -> np.array:
        """Compute the fraction of trajectories in the regular phase."""
        # Comput local maximum
        wbarrier = (model_config.w0 * model_config.d1 - model_config.w0 * model_config.d2) / (model_config.d1 + model_config.d2)
        # Delete exploding trajectories
        phases = np.zeros_like(trajectories)
        phases[trajectories - wbarrier > 0] = 1
        phases[trajectories - wbarrier < 0] = -1
        # Plot fraction of trajectories escaping from regular to singular phase
        return np.sum(phases == -1, axis=0) / phases.shape[0] + 1e-8

    def _compute_escape_rate(
        self,
        fraction: np.array,
        tmin: int = 3,
        frac_max: float = 10**-3,
    ) -> tuple[np.array, float]:
        """Compute the escape rate from the fraction of trajectories in the non-degenerate minimum."""
        # Find where fraction drops below frac_max
        tmax = np.argmax(fraction < frac_max)
        if tmax == 0:
            # If no values are below frac_max, use the entire array
            tmax = len(fraction)
        
        # Ensure we have enough points for regression
        if tmax <= tmin:
            # Return default values if we don't have enough points
            return 0.0, 0.0
            
        time = np.arange(0, len(fraction))
        regress_time = np.arange(tmin, tmax, 1)
        
        # Ensure we have data to regress
        if len(regress_time) == 0:
            return 0.0, 0.0
            
        regress_log_frac = np.log(fraction[tmin:tmax])
        X_with_const = sm.add_constant(regress_time)
        
        try:
            model = sm.OLS(regress_log_frac, X_with_const).fit()
            slope = model.params[1]
            escape_rate = np.abs(slope) 
            error = model.bse[1]
            
            # Generate predictions and plot
            predictions = model.get_prediction(X_with_const)
            predictions_summary = predictions.summary_frame(alpha=0.05)
            title_lines = [
                f"Fraction of trajectories in the non-degenerate minimum",
                rf"$B={self.trainer_config.batch_size}$, $\eta={self.trainer_config.lr}$, $w_0={self.model_config.w0:.2f}$",
                # f"Escape rate is {escape_rate:.2e}"
            ]
            title = "\n".join(title_lines)
            plt.figure()
            plt.scatter(time, fraction, label="fraction", marker="x", color="orange")
            plt.plot(
                regress_time,
                np.exp(predictions_summary["mean"]),
                label="regression",
                color="purple",
            )
            plt.fill_between(
                regress_time,
                np.exp(predictions_summary["obs_ci_lower"]),
                np.exp(predictions_summary["obs_ci_upper"]),
                alpha=0.5,
                label="95% CI",
            )
            plt.xlabel("Time")
            plt.ylabel("Fraction")
            plt.yscale("log")
            plt.ylim((10**-3, 1))
            plt.legend()
            plt.title(title)
            fname = f"regression_B_{self.trainer_config.batch_size}_lr_{self.trainer_config.lr}_w0_{self.model_config.w0:.2e}.png"
            fpath = self.trainer_config.output_dir.joinpath(fname)
            plt.savefig(fpath)
            return escape_rate, error
        
        except Exception as e:
            print(f"Regression failed: {e}")
            return 0.0, 0.0


    def _compute_and_store_results(
        self, 
        escape_dict: dict,
        clean_traj: np.ndarray,
        model: PolyModel,
        params: dict
    ) -> None:
        """Compute escape rate and store results."""
        fraction = self._regular_fraction(clean_traj, model)
        escape_rate, error = self._compute_escape_rate(
            fraction,
            frac_max=params['frac_max'],
            tmin=params['tmin'],
        )
        
        # Store results
        escape_dict["escape_rate"].append(escape_rate)
        escape_dict["lr"].append(params['lr'])
        escape_dict["B"].append(params['batch_size'])
        escape_dict["w0"].append(params['w0'])
        escape_dict["error"].append(error)
        escape_dict["fraction"].append(fraction)

    def _save_results(self, df: pd.DataFrame, frac_max: float) -> None:
        """Save results to CSV file."""
        fname = (f"escape_rate_to_{frac_max}_trajectories_"
                f"nSGD_{self.trainer_config.nSGD}_"
                f"nsamples_{self.trainer_config.nsamples}.csv")
        fpath = self.trainer_config.output_dir.joinpath(fname)
        df.to_csv(fpath)

    def parameter_sweep(
        self,
        w0_range: np.ndarray,
        batch_range: np.ndarray,
        lr_range: np.ndarray,
        model: PolyModel,
        tmin: int = 3,
        frac_max: float = 10**-2
    ) -> pd.DataFrame:
        """
        Run multiple experiments with varying parameters and compute escape rate for each combination.
        
        Args:
            w0_range: Range of initial weights to test
            batch_range: Range of batch sizes to test
            lr_range: Range of learning rates to test
            model: Model to train
            tmin: Minimum time threshold
            frac_max: Maximum fraction threshold
        
        Returns:
            DataFrame containing results for all parameter combinations
        """
        total_experiments, escape_dict = self._setup_experiment_tracking(
            w0_range, batch_range, lr_range
        )
        
        for exp_idx, (w0, batch_size, lr) in enumerate(
            itertools.product(w0_range, batch_range, lr_range)
        ):
            if exp_idx % 10 == 0:
                print(f"Running experiment {exp_idx} over {total_experiments}")
            
            # Update parameters
            model.update_params(w0=w0)
            self.trainer_config.batch_size = int(batch_size)
            self.trainer_config.lr = lr
            
            # Generate and process trajectories
            df = self.trajectory_generator.generate(model)
            clean_traj = self._process_trajectories(df)
            
            # Compute and store results
            params = {
                'frac_max': frac_max,
                'tmin': tmin,
                'batch_size': batch_size,
                'lr': lr,
                'w0': w0
            }
            self._compute_and_store_results(escape_dict, clean_traj, model, params)
        
        results_df = pd.DataFrame.from_dict(escape_dict)
        self._save_results(results_df, frac_max)
        
        return results_df