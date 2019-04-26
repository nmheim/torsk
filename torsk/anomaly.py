import numpy as np
from scipy import special

def cumulative_distribution(x):
    return special.erf(x/2.**.5)

def qfunction(x):
    return 1 - cumulative_distribution(x)

def sliding_score(error, small_window, large_window):
    scores = np.zeros(error.shape)
    lw_mu = np.zeros_like(scores)
    lw_std = np.zeros_like(scores)
    sw_mu = np.zeros_like(scores)

    for i in range(small_window, error.shape[0]):
        end = i + small_window
        sw_start = max(0, end - small_window)
        lw_start = max(0, end - large_window)

        sw_err = error[sw_start:end]
        lw_err = error[lw_start:end]

        lw_mu[i] = lw_err.mean(axis=0)
        lw_std[i] = lw_err.std(axis=0)
        sw_mu[i] = sw_err.mean(axis=0)

        x = np.abs(lw_mu[i] - sw_mu[i]) / lw_std[i]
        s = qfunction(x)
        scores[i] = s

    return scores, lw_mu, lw_std, sw_mu
