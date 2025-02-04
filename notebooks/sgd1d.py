#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import ast
import os
import sys
import pymc as pm
import pandas as pd

from pathlib import Path

from lib.models.models import PolyModel
from lib.config.model_config import PolyModel1DConfig, TrainerConfig
from lib.training.trainer import SGDTrainer
from lib.training.trajectory_generator import TrajectoryGenerator
from lib.training.trajectory_generator import ParameterSweeper
from lib.utils.visualization import plot_potential


#%%

# Plotting settings
plt.style.use('ggplot')
fig_width = 25  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
label_size = 22
tick_size = 20
params = {'backend': 'ps',
          'lines.linewidth': 1.5,
          'axes.labelsize': label_size,
          'axes.titlesize': label_size,
          'font.size': label_size,
          'legend.fontsize': tick_size,
          'xtick.labelsize': tick_size,
          'ytick.labelsize': tick_size,
          'text.usetex': False,
          'figure.figsize': fig_size,
          "font.weight": "bold",
          "axes.labelweight": "bold"}
plt.rcParams.update(params)

#%%

model_cfg = PolyModel1DConfig()
nSGD = 10**2
nsamples = 10**2
trainer_cfg = TrainerConfig(nSGD=nSGD, nsamples=nsamples, output_dir=Path("../data/"))

frac_max = 10**-3
#%%

model = PolyModel(model_cfg)
trainer = SGDTrainer(trainer_cfg)
trajectory_generator = TrajectoryGenerator(model_config=model_cfg, trainer_config=trainer_cfg)
parameter_sweeper = ParameterSweeper(model_config=model_cfg, trainer_config=trainer_cfg)

#%% Compute and plot escape rate
%matplotlib inline
df = trajectory_generator.generate(model)
clean_trajectories = parameter_sweeper._process_trajectories(df)
regular_fraction = parameter_sweeper._regular_fraction(clean_trajectories, model_cfg)
escape_rate = parameter_sweeper._compute_escape_rate(fraction=regular_fraction, frac_max=frac_max, tmin=3)
#%% Plot potential
plot_potential(model, nsamp=10**4, ymax=500)

# %% Density of SGD trajectories

wf = []
for i in range(len(df)):
    wf.append(df["trajectory"][i][-1])
niterations = trainer_cfg.nSGD/trainer_cfg.batch_size
plt.hist(wf, bins=50, density=True, color="purple")
plt.xlabel("End of trajectory")
plt.ylabel("Density")
plt.xlim((-model_cfg.wmax, model_cfg.wmax))
plt.title(f"Distribution of SGD trajectories after {niterations} iterations")
# %% Bayesian posterior
# Exact
n_values = [1, 100, 1000]

w = np.linspace(-4, 4, 400)
K_w = (w - model_cfg.w0)**4 * (w + model_cfg.w0)**2
# Plotting the function for different values of n
plt.figure(figsize=(12, 8))

for n in n_values:
    l = np.exp(-n*K_w)
    plt.plot(w, l, label=f'n = {n}')
plt.title('Bayesian posterior for n samples')
plt.xlabel('w')
plt.xlim((-model_cfg.wmax, model_cfg.wmax))
plt.ylabel('Posterior')
plt.legend()
plt.grid(True)
plt.yscale("linear")  # Using a logarithmic scale for better visualization
plt.show()

# %%
