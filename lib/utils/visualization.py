import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from pathlib import Path

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
    
    return model.params[1], model.bse[1]  # Return escape rate and error 