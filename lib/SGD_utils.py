import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import copy
import ast
import statsmodels.api as sm

from pandas import DataFrame
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


# Models
class TrivialModel(torch.nn.Module):
    """
    Trivial, single parameter model
    """

    def __init__(self) -> None:
        super(TrivialModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn([1]))

    def forward(self, input: Tensor):
        return input * 0.0


# Defining the Linear Model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # Random weight initialization
        self.linear.weight.data.normal_()

    def forward(self, x):
        return self.linear(x)


class PolyModel(torch.nn.Module):
    def __init__(
        self,
        w0: float = 2,
        wmin=-4,
        wmax=4,
        d1: int = 1,
        d2: int = 2,
        in_features: int = 1,
        out_features: int = 1,
        seed: int = 1,
        w_init: Optional[Tensor] = None,
    ) -> None:
        super(PolyModel, self).__init__()
        self.w0 = w0
        self.d1 = d1
        self.d2 = d2
        self.wmin = wmin  # boundary of initialization
        self.wmax = wmax
        self.seed = seed  # Seed for initialisation of trajectories
        self.w_init = w_init
        self.in_features = in_features
        self.out_features = out_features
        # Define the weight parameter
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        if self.w_init is not None:
            self.weight = torch.nn.Parameter(w_init)
        else:
            torch.nn.init.uniform_(self.weight, self.wmin, self.wmax)

    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def forward(self, input: Tensor):
        w1 = (self.weight + self.w0) ** self.d1
        w2 = (self.weight - self.w0) ** self.d2
        return input * w1 * w2

    def gradient(self):
        w1 = (self.weight + self.w0) ** self.d1
        dw1 = self.d1 * (self.weight + self.w0) ** (self.d1 - 1)
        w2 = (self.weight - self.w0) ** self.d2
        dw2 = self.d2 * (self.weight - self.w0) ** (self.d2 - 1)
        return dw1 * w2 + w1 * dw2


class PolyModel2D(torch.nn.Module):
    def __init__(
        self,
        wxm=4,
        wym=4,
        d: int = 2,
        in_features: int = 1,
        out_features: int = 1,
        seed: int = 1,
        wx_init: Optional[Tensor] = None,
        wy_init: Optional[Tensor] = None,
    ) -> None:
        super(PolyModel2D, self).__init__()
        self.d = d
        self.wxmin = -wxm  # boundary of initialization
        self.wxmax = wxm  # boundary of initialization
        self.wymin = -wym
        self.wymax = wym
        self.seed = seed  # Seed for initialisation of trajectories
        self.wx_init = wx_init
        self.wy_init = wy_init
        self.in_features = in_features
        self.out_features = out_features

        # Define the weight parameters
        self.weight1 = torch.nn.Parameter(torch.empty((out_features, in_features)))
        self.weight2 = torch.nn.Parameter(torch.empty((out_features, in_features)))
        self.weight = torch.stack([self.weight1, self.weight2])
        if self.wx_init is not None:
            self.weight1 = torch.nn.Parameter(wx_init)
            self.weight2 = torch.nn.Parameter(wy_init)
        else:
            torch.nn.init.uniform_(self.weight1, self.wxmin, self.wxmax)
            torch.nn.init.uniform_(self.weight2, self.wymin, self.wymax)

    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def forward(self, input: Tensor):
        w1 = self.weight1
        w2 = self.weight2

        Q = (w1**2 + w2**2) ** 2 * w1**self.d

        return input * Q


# %% Experiments
class SGDPolyRunner:
    def __init__(
        self,
        nSGD=10**4,
        nsamples=10**3,
        batch_size=30,
        lr=0.01,
        momentum=0,
        auto=True,
        seed=1,
        shuffle=True,
    ):
        self.nSGD = nSGD
        self.nsamples = nsamples
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.auto = auto  # automatic or analytic gradient
        self.seed = seed
        self.shuffle = shuffle

    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def make_dataset(self, model: PolyModel) -> DataLoader:
        torch.manual_seed(self.seed)
        x_data = torch.randn((self.nsamples, model.out_features, model.in_features))
        y_data = torch.randn((self.nsamples, model.out_features, model.in_features))
        dataset = TensorDataset(x_data, y_data)
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
        return data_loader

    def sgd_step(self, model: PolyModel, xb, yb):
        """
        Perform an SGD step using the manually computed gradient.
        """
        # Compute the analytic gradient
        grad = model.gradient()

        # Compute the loss terms
        predicted = model.forward(xb)  # Forward pass to get predictions
        error = predicted - yb
        loss_grad = 2 * xb * error / xb.size(0)  # d(loss)/d(prediction), for MSE loss
        # Update the weights
        # Note: The update rule is weight = weight - learning_rate * d(loss)/d(weight)
        #       And d(loss)/d(weight) is obtained using the chain rule: d(loss)/d(prediction) * d(prediction)/d(weight)
        update = model.weight - self.lr * grad * torch.sum(loss_grad, dim=0)
        model.update_params(weight=torch.nn.Parameter(update))
        return grad

    def train(self, model: PolyModel, dataset: DataLoader) -> tuple:
        """
        Train with pytorch automatic differentiation to compute gradient
        """
        # Loss and weights tracking
        running_loss = []
        w_init_copy = copy.deepcopy(model.w_init.clone())
        if model.w_init == None:
            running_weight = []
        else:
            running_weight = [w_init_copy.item()]
        model.double()
        # Loss and Optimizer
        loss_function = nn.MSELoss()
        if self.auto == True:
            optimizer = optim.SGD(
                model.parameters(), momentum=self.momentum, lr=self.lr
            )
        # Training the Model
        for xb, yb in dataset:
            xb = xb.double()
            yb = yb.double()
            # Forward pass
            if self.auto == True:
                y_pred = model.forward(xb)
                loss = loss_function(y_pred, yb)
                running_loss.append(loss.item())
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                self.sgd_step(model, xb, yb)
            running_weight.append(model.weight.item())
        return running_loss, running_weight

    def train2d(self, model: PolyModel2D, dataset: DataLoader) -> tuple:
        """
        Train with pytorch automatic differentiation to compute gradient
        """
        # Loss and weights tracking
        running_loss = []
        if model.wx_init == None:
            running_weight1 = []
            running_weight2 = []
        else:
            w1_init_copy = model.weight1.clone()
            w2_init_copy = model.weight2.clone()
            running_weight1 = [w1_init_copy.item()]
            running_weight2 = [w2_init_copy.item()]
        model.double()
        # Loss and Optimizer
        loss_function = nn.MSELoss()
        if self.auto == True:
            optimizer = optim.SGD(
                model.parameters(), momentum=self.momentum, lr=self.lr
            )
        # Training the Model
        for xb, yb in dataset:
            xb = xb.double()
            yb = yb.double()
            # Forward pass
            if self.auto == True:
                y_pred = model.forward(xb)
                loss = loss_function(y_pred, yb)
                running_loss.append(loss.item())
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_weight1.append(model.weight1.item())
            running_weight2.append(model.weight2.item())
        return running_loss, running_weight1, running_weight2

    def generate_trajectories(self, model: PolyModel) -> DataFrame:
        # Set up seed for reproducibility
        np.random.seed(seed=self.seed)
        data = {"w_init": [], "trajectory": [], "loss": []}
        dataset = self.make_dataset(model)
        w_inits = np.random.uniform(
            model.wmin,
            model.wmax,
            size=(model.out_features, model.in_features, self.nSGD),
        )
        for i in range(self.nSGD):
            if i + 1 % 1000 == 0:
                print(f"trajectory {i} over {self.nSGD}")
            wi = torch.tensor(w_inits[:, :, i])
            model.update_params(w_init=wi)
            model.update_params(
                weight=torch.nn.Parameter(model.w_init)
            )  # re-initialise before training
            running_loss, running_weight = self.train(model, dataset)
            data["w_init"].append(model.w_init.item())
            data["trajectory"].append(running_weight)
            data["loss"].append(running_loss)
        df = pd.DataFrame(data)
        fpath = Path("../data/")
        # Check if the folder exists
        if not fpath.exists() or not fpath.is_dir():
            raise FileNotFoundError(f"The folder '{fpath}' does not exist.")
        model_params = (
            f"w0_{model.w0}_d1_{model.d1}_d2_{model.d2}_seed_{model.seed}.csv"
        )
        sgd_params = f"lr_{self.lr}_b_{self.batch_size}_seed_{self.seed}_N_{self.nSGD}_m_{self.nsamples}"
        fname = "sgd_traj"
        fname = "_".join([fname, sgd_params, model_params])
        fpath = fpath.joinpath(fname)
        df.to_csv(fpath)
        return df

    def generate_trajectories2d(self, model: PolyModel2D) -> DataFrame:
        # Set up seed for reproducibility
        np.random.seed(seed=self.seed)
        data = {
            "w1_init": [],
            "w2_init": [],
            "trajectory1": [],
            "trajectory2": [],
            "loss": [],
        }
        dataset = self.make_dataset(model)
        wx_inits = np.random.uniform(
            model.wxmin,
            model.wxmax,
            size=(model.out_features, model.in_features, self.nSGD),
        )
        wy_inits = np.random.uniform(
            model.wymin,
            model.wymax,
            size=(model.out_features, model.in_features, self.nSGD),
        )

        for i in range(self.nSGD):
            if i + 1 % 1000 == 0:
                print(f"trajectory {i} over {self.nSGD}")
            wxi = torch.tensor(wx_inits[:, :, i])
            wyi = torch.tensor(wy_inits[:, :, i])
            model.update_params(wx_init=wxi, wy_init=wyi)
            model.update_params(
                weight1=torch.nn.Parameter(model.wx_init),
                weight2=torch.nn.Parameter(model.wy_init),
            )  # re-initialise before training
            running_loss, running_weight1, running_weight2 = self.train2d(
                model, dataset
            )
            data["w1_init"].append(model.weight1.item())
            data["w2_init"].append(model.weight2.item())
            data["trajectory1"].append(running_weight1)
            data["trajectory2"].append(running_weight2)
            data["loss"].append(running_loss)
        df = pd.DataFrame(data)
        fpath = Path("../data/")
        # Check if the folder exists
        if not fpath.exists() or not fpath.is_dir():
            raise FileNotFoundError(f"The folder '{fpath}' does not exist.")
        model_params = f"d_{model.d}_seed_{model.seed}.csv"
        sgd_params = f"lr_{self.lr}_b_{self.batch_size}_seed_{self.seed}_N_{self.nSGD}_m_{self.nsamples}"
        fname = "sgd_traj_2D"
        fname = "_".join([fname, sgd_params, model_params])
        fpath = fpath.joinpath(fname)
        df.to_csv(fpath)
        return df

    def parameter_sweep(
        self, w0_range, batch_range, lr_range, model: PolyModel, tmin=3, frac_max=10**-2
    ):
        """
        Run multiple experiments with varying parameters and compute escape rate for each params
        """
        iexp = 0
        nexp = len(w0_range) * len(batch_range) * len(lr_range)
        escape_dict = {
            "escape_rate": [],
            "lr": [],
            "B": [],
            "w0": [],
            "fraction": [],
            "error": [],
        }
        for w0 in w0_range:
            model.update_params(w0=w0)
            for batch_size in batch_range:
                self.update_params(batch_size=int(batch_size))
                for lr in lr_range:
                    if iexp % 10 == 0:
                        print(f"Running experiment {iexp} over {nexp}")
                    self.update_params(lr=lr)
                    df = self.generate_trajectories(model)
                    trajectories = np.asarray(df["trajectory"].to_list())
                    clean_traj = trajectories[~np.isnan(trajectories).any(axis=1)]
                    fraction = regular_fraction(clean_traj, model)
                    escape_rate, error = compute_escape_rate(
                        fraction,
                        frac_max=frac_max,
                        tmin=tmin,
                        batch_size=batch_size,
                        lr=lr,
                        w0=w0,
                    )
                    escape_dict["escape_rate"].append(escape_rate)
                    escape_dict["lr"].append(lr)
                    escape_dict["B"].append(batch_size)
                    escape_dict["w0"].append(w0)
                    escape_dict["error"].append(error)
                    escape_dict["fraction"].append(fraction)
                    iexp += 1
        df = pd.DataFrame.from_dict(escape_dict)
        fname = f"escape_rate_to_{frac_max}_trajectories_nSGD_{self.nSGD}_nsamples_{self.nsamples}.csv"
        fpath = Path("../data/regressions")
        fpath = fpath.joinpath(fname)
        df.to_csv(fpath)
        return df


def compute_likelihood(model: PolyModel, x, y):
    y_pred = model(x)
    mse_loss = torch.nn.MSELoss()(y_pred, y)
    return torch.exp(-mse_loss)


def prior_prob(model):
    # Define your prior probability distribution for the model parameters
    # For example, you can use a uniform prior within the specified bounds
    if model.wmin <= model.weight <= model.wmax:
        return 1.0
    else:
        return 0.0


def metropolis_hastings(model, x, y, num_samples=1000, step_size=0.1):
    samples = []
    current_weight = model.weight.clone()

    for _ in range(num_samples):
        # Propose a new weight
        proposed_weight = current_weight + torch.randn_like(current_weight) * step_size

        # Create a new model with the proposed weight
        proposed_model = PolyModel(w_init=proposed_weight)

        # Compute the likelihood and prior probabilities
        current_likelihood = compute_likelihood(model, x, y)
        proposed_likelihood = compute_likelihood(proposed_model, x, y)
        current_prior = prior_prob(model)
        proposed_prior = prior_prob(proposed_model)

        # Compute the acceptance probability
        acceptance_prob = min(
            1,
            (proposed_likelihood * proposed_prior)
            / (current_likelihood * current_prior),
        )

        # Accept or reject the proposed weight
        if torch.rand(1) < acceptance_prob:
            current_weight = proposed_weight
            model.weight.data = proposed_weight

        samples.append(current_weight.clone())

    return samples


def theoretical_loss(model: PolyModel, w, x, y):
    """
    Compute theoretical as the empirical loss at parameter w over all data samples x and y
    """
    loss_function = nn.MSELoss()
    model.update_params(weight=torch.nn.Parameter(torch.tensor(w)))
    y_pred = model.forward(x)
    return loss_function(y_pred, y).item()


def regular_fraction(trajectories: np.array, model: PolyModel) -> np.array:
    # Comput local maximum
    wbarrier = (model.w0 * model.d1 - model.w0 * model.d2) / (model.d1 + model.d2)
    # Delete exploding trajectories
    phases = np.zeros_like(trajectories)
    phases[trajectories - wbarrier > 0] = 1
    phases[trajectories - wbarrier < 0] = -1
    # Plot fraction of trajectories escaping from regular to singular phase
    return np.sum(phases == -1, axis=0) / phases.shape[0]


def is_literal_eval_successful(s):
    try:
        ast.literal_eval(s)
        return True
    except ValueError:
        return False


def read_clean_trajectories(fpath: Path):
    """
    Filter exploding trajectories and convert strings into array of trajectories
    """
    df = pd.read_csv(fpath)
    df = df[df["trajectory"].apply(is_literal_eval_successful)]
    df["trajectory"] = df["trajectory"].apply(ast.literal_eval)
    trajectories = np.asarray(df["trajectory"].to_list())
    clean_traj = trajectories[~np.isnan(trajectories).any(axis=1)]
    return clean_traj


def compute_escape_rate(
    fraction, tmin=3, frac_max=10**-3, batch_size=20, lr=0.01, w0=1.5
):
    tmax = np.argmax(fraction < frac_max)
    if tmax == 0:
        tmax = len(fraction)
    time = np.arange(0, len(fraction))
    regress_time = np.arange(tmin, tmax, 1)
    log_frac = np.log(fraction)
    regress_log_frac = np.log(fraction[tmin:tmax])  # Drop outliers for the regression
    # stats = linregress(regress_time, regress_log_frac)
    X_with_const = sm.add_constant(regress_time)
    model = sm.OLS(regress_log_frac, X_with_const).fit()
    escape_rate = np.abs(model.params[1])
    error = model.bse[1]
    predictions = model.get_prediction(X_with_const)
    predictions_summary = predictions.summary_frame(
        alpha=0.05
    )  # 95% confidence interval
    title_lines = [
        f"Fraction of trajectories in the non-degenerate minimum",
        rf"$B={batch_size}$, $\eta={lr}$, $w_0={w0:.2f}$",
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
    fname = f"regression_B_{batch_size}_lr_{lr}_w0_{w0:.2e}.png"
    fpath = Path("../data/")
    fpath = fpath.joinpath(fname)
    plt.savefig(fpath)
    return escape_rate, error


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
    trajectories, model: PolyModel, sgd_runner: SGDPolyRunner, yscale="log"
):
    fraction = regular_fraction(trajectories, model)
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
