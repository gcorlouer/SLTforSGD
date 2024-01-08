import numpy as np
import torch 
import pandas as pd
import os 


from pandas import DataFrame
from pathlib import Path
from sys import path
path.insert(0, "lib")
from lib.onedmodel import * 
# Get the directory where the script is located
script_directory = Path(__file__).resolve().parent

# Change the current working directory to the script directory
os.chdir(script_directory)
print(script_directory)

# Models
class TrivialModel(torch.nn.Module):
    """
    Trivial, single parameter model
    """
    def __init__(self) -> None:
        super(TrivialModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn([1]))

    def forward(self, input:Tensor):
        return input * 0.

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
    def __init__(self, w0:float=2, d1:int=1, d2:int=2, in_features:int=1,
                 out_features:int=1, w_init: Optional[Tensor] = None) -> None:
         super(PolyModel, self).__init__()
         self.w0 = w0
         self.d1 = d1
         self.d2 = d2
         if w_init is not None:
            self.weight = torch.nn.Parameter(w_init)
         else:
            self.weight = torch.nn.Parameter(torch.randn((out_features, in_features)))

    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def forward(self, input:Tensor):
        # Chose cst1 and cst2 such that K(1) = 0 and K'(1) = cst
        w1 = (self.weight + self.w0)**self.d1
        w2 = (self.weight - self.w0)**self.d2
        return input * w1 * w2

# SGD
class SGDPoly:
    def __init__(self, nSGD=10**4, num_samples=10**3, batch_size=30, lr=0.01, 
                 momentum=0, num_epochs=10, linear=False, shuffle=True):
        self.nSGD = nSGD
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.linear = linear
        self.shuffle = shuffle


    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


    def make_dataset(self) -> DataLoader:
        x_data = torch.normal(0., 1., (self.num_samples, 1))
        y_data = torch.normal(0., 1., (self.num_samples, 1))
        dataset = TensorDataset(x_data, y_data)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return data_loader


    def train_model(self, model: PolyModel, data_loader : DataLoader, w_init: Optional[Tensor] = None) -> tuple:
        # Loss tracking
        running_loss = []

        # Tracking weights
        if w_init is None:
            running_weight = []
        else:
            running_weight = [w_init.item()]

        # Loss and Optimizer
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), momentum=self.momentum, lr=self.lr)

        # Training the Model
        for epoch in range(self.num_epochs):
            for batch_x, batch_y in data_loader:
                # Forward pass
                y_pred = model(batch_x)
                loss = loss_function(y_pred, batch_y)
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
                if self.linear == True:
                    current_weight = model.linear.weight.item()
                    running_weight.append(current_weight)
                else:
                    current_weight = model.weight.item() 
                    running_weight.append(current_weight)
        return running_loss, running_weight
    

    def sgd_on_poly(self, model: PolyModel) -> DataFrame:
        wm = 2 * model.w0
        w_inits = np.random.uniform(-wm, wm, size=self.nSGD)
        data = {'w_init': [], 'trajectory': [], 'loss': []}
        for i in range(self.nSGD):
            if i % 1000 == 0:
                print(f"trajectory {i} over {self.nSGD}")
            wi = torch.tensor(w_inits[i])
            model.update_params(w_init=wi)
            data_loader = make_dataset(num_samples=self.num_samples, batch_size=self.batch_size)
            running_loss, running_weight = self.train_model(model, data_loader, lr=self.lr, 
                                                    momentum=self.momentum, num_epochs=self.num_epochs,
                                                    w_init=wi, linear=self.linear)
            data['w_init'].append(wi.item())
            data['trajectory'].append(running_weight)
            data['loss'].append(running_loss)

        # Construct the filename
        fpath = Path('../data')
        filename = f"experiment_d1_{model.d1}_d2_{model.d2}_w0_{model.w0}_lr_{self.lr}_momentum_{self.momentum}_batch_{self.batch_size}_epochs_{self.num_epochs}_nsamples_{self.num_samples}.csv"
        df = pd.DataFrame(data)
        fpath = fpath.joinpath(filename)
        df.to_csv(fpath, index=False)
        return df


    def multi_sgd_on_poly(self, w0_range, batch_range, lr_range, model: PolyModel):
        """
        Run multiple experiments with varying parameters.
        """
        for w0 in w0_range:
            model.update_params(w0=w0)
            for batch_size in batch_range:
                self.update_params(batch_size=int(batch_size))
                for lr in lr_range:
                    self.update_params(lr=lr)
                    self.sgd_on_poly(model)
    

#%%
def run_multiple_experiments(w0_range, batch_range, lr_range,
                             nSGD: int = 10**4, d1: int = 1, d2: int = 1, 
                             num_samples: int = 10**3, momentum: float = 0, 
                             num_epochs: int = 10, linear: bool = False) -> None:
    """
    Run multiple experiments with varying parameters.
    """
    for w0 in w0_range:
        for batch_size in batch_range:
            batch_size = int(batch_size)
            for lr in lr_range:
                run_experiment(nSGD, d1=d1, d2=d2, w0=w0, num_samples=num_samples,
                               lr=lr, momentum=momentum, num_epochs=num_epochs, 
                               batch_size=batch_size, linear=linear)

    
    

def run_experiment(nSGD, d1: int = 1, d2: int = 1, w0: float = 1, 
                   num_samples: int = 10**3, lr: float = 0.01, 
                   momentum: float = 0, num_epochs: int = 10, 
                   batch_size: int = 30, linear: bool = False) -> DataFrame:
    """
    Run a single experiment with specified parameters.

    Returns:
        DataFrame: A DataFrame containing the initial weights, trajectories, and loss values.
    """
    wm = 2 * w0
    w_inits = np.random.uniform(-wm, wm, size=nSGD)
    data = {'w_init': [], 'trajectory': [], 'loss': []}

    for i in range(nSGD):
        if i % 1000 == 0:
            print(f"trajectory {i} over {nSGD}")
        wi = torch.tensor(w_inits[i])
        model = PolyModel(w0, d1=d1, d2=d2, in_features=1, out_features=1, w_init=wi)
        data_loader = make_dataset(num_samples=num_samples, batch_size=batch_size)
        running_loss, running_weight = train_model(model, data_loader, lr=lr, 
                                                   momentum=momentum, num_epochs=num_epochs,
                                                   w_init=wi, linear=linear)
        data['w_init'].append(wi.item())
        data['trajectory'].append(running_weight)
        data['loss'].append(running_loss)

    # Construct the filename
    fpath = Path('../data')
    filename = f"experiment_d1_{d1}_d2_{d2}_w0_{w0}_lr_{lr}_momentum_{momentum}_batch_{batch_size}_epochs_{num_epochs}_nsamples_{num_samples}.csv"
    df = pd.DataFrame(data)
    fpath = fpath.joinpath(filename)
    df.to_csv(fpath, index=False)
    return df


#%%

w0_range = np.arange(1, 3, 0.5)
batch_range = np.arange(20, 30, 5)
lr_range = [0.1, 0.01, 0.001]
d1 = 1
d2 = 2
num_samples = 10 ** 3
momentum = 0
num_epochs = 10
nSGD = 10**4

run_multiple_experiments(w0_range, batch_range, lr_range,
                            nSGD=nSGD, d1=d1, d2=d2, num_samples=num_samples,
                            momentum=momentum, num_epochs=num_epochs, linear=False)


