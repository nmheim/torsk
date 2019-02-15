import numpy as np

def cumulative_distribution(x):
    return np.erf(x/2.**.5)

def qfunction(x):
    return 1 - cumulative_distribution(x)

def sliding_score(error, small_window, large_window):
    assert len(error.shape) == 1

    shape = [error.shape[0] - large_window,]
    scores = np.zeros(shape)

    for i in range(shape[0]):
        j = i + large_window
        sw_err = error[j:j+small_window]
        lw_err = error[j-large_window:j+small_window]
        mu = lw_err.mean(axis=0)
        std = lw_err.std(axis=0)
        sw_mu = sw_err.mean(axis=0)
        x = np.abs(mu - sw_mu) / std
        s = qfunction(x)
        scores[i] = s
    return scores
