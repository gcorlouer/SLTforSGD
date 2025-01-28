import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import torch

from pathlib import Path
from lib.models.models import PolyModel
from lib.training.trainer import SGDTrainer
from lib.training.trajectory_generator import TrajectoryGenerator, ParameterSweeper

def plot_escape_rate(fraction, config, tmin=3, frac_max=10**-2):
    """Plot escape rate analysis."""
    tmax = np.argmax(fraction < frac_max)
    if tmax == 0:
        tmax = len(fraction)
        
    time = np.arange(0, len(fraction))
    regress_time = np.arange(tmin, tmax, 1)
    regress_log_frac = np.log(fraction[tmin:tmax])
    
    X_with_const = sm.add_constant(regress_time)
    model = sm.OLS(regress_log_frac, X_with_const).fit()
    
    predictions = model.get_prediction(X_with_const)
    predictions_summary = predictions.summary_frame(alpha=0.05)
    
    plt.figure()
    plt.scatter(time, fraction, label="fraction", marker="x", color="orange")
    plt.plot(regress_time, np.exp(predictions_summary["mean"]), 
            label="regression", color="purple")
    plt.fill_between(regress_time,
                    np.exp(predictions_summary["obs_ci_lower"]),
                    np.exp(predictions_summary["obs_ci_upper"]),
                    alpha=0.5, label="95% CI")
    
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.yscale("log")
    plt.ylim((10**-3, 1))
    plt.legend()
    plt.title(f"Escape Rate Analysis\nB={config.batch_size}, η={config.lr}")

def theoretical_loss(model: PolyModel, w, x, y):
    """
    Compute theoretical as the empirical loss at parameter w over all data samples x and y
    """
    loss_function = nn.MSELoss()
    model.update_params(weight=torch.nn.Parameter(torch.tensor(w)))
    y_pred = model.forward(x)
    return loss_function(y_pred, y).item()

def theoretical_loss2d(model, nsamples, wxm, wym):
    num_points = 1000
    x = torch.randn((nsamples, 1, 1))
    y = torch.randn((nsamples, 1, 1))
    wx_values = np.linspace(-wxm, wxm, num_points)
    wy_values = np.linspace(-wym, wym, num_points)
    wx, wy = np.meshgrid(wx_values, wy_values)
    w = np.stack((wx, wy), axis=-1)

    loss = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            loss[i, j] = theoretical_loss(model, w[i, j], x, y)

def plot_potential(model: PolyModel, nsamp=10**4, ymax=500):
    wbarrier = (model.w0 * model.d1 - model.w0 * model.d2) / (model.d1 + model.d2)
    a = float(model.w0)
    x = torch.randn((nsamp, 1, 1))
    y = torch.randn((nsamp, 1, 1))
    wrange = np.linspace(model.wmin, model.wmax, 1000)
    loss = np.zeros((len(wrange), 1))
    for i in range(len(wrange)):
        loss[i] = theoretical_loss(model, wrange[i], x, y)

    min1 = theoretical_loss(model, -a, x, y)
    min2 = theoretical_loss(model, a, x, y)
    # Split the data at x = b
    regular = wrange < wbarrier
    singular = wrange >= wbarrier

    plt.plot(wrange[regular], loss[regular], color="purple", label="Sharp")
    plt.plot(wrange[singular], loss[singular], color="orange", label="Broad")
    plt.xlabel("Weight")
    plt.ylabel("Potential")
    plt.axvline(x=wbarrier, linestyle="--", color="k", label="barrier")
    plt.scatter(
        [-a, a], [min1, min2], color="blue", s=100, zorder=5, marker="x", label="Minima"
    )
    plt.legend()
    plt.ylim((0, ymax))
    plt.title("Potential")
    plt.show()


def plot_fraction_traj(
    trajectories, model: PolyModel, generator: TrajectoryGenerator, trainer: SGDTrainer, yscale="log"
):
    sweeper = ParameterSweeper(model.config, trainer.config)
    fraction = sweeper._regular_fraction(trajectories, model)
    nit = len(fraction)
    time = np.arange(0, nit, 1)
    plt.plot(time, fraction[time], color="purple")
    plt.xlabel("Time")
    plt.ylabel("fraction")
    plt.yscale(yscale)
    plt.title(
        "Fraction of trajectories in the low degenerate phase \n"
        f"b: {sgd_runner.batch_size}, lr:{sgd_runner.lr}, N:{sgd_runner.nSGD}, m: {sgd_runner.nsamples}"
    )
    plt.ylim((0.001, 1))


def plot_contour_2d(loss, df, wym, wx, wy):
    trajx = df["trajectory1"][0]
    trajy = df["trajectory2"][0]
    fig, ax = plt.subplots()
    plt.contourf(wx, wy, loss, levels=20, cmap="viridis")
    plt.colorbar(label="Theoretical Loss")
    plt.plot(trajx, trajy, linestyle="--", color="purple", label="trajectory")
    plt.xlabel(r"$w_1$")
    plt.ylabel(r"$w_2$")
    plt.xlim((-1, 1))
    plt.ylim((-wym, wym))
    plt.grid(True)
    plt.scatter(0, 0, marker="x", label="Degenerate point")
    plt.scatter(wx[0], wy[0], marker="x", label="start", color="blue")
    plt.axvline(x=0, linestyle="--", label="Degenerate line", color="k")
    plt.legend()
    plt.show()
