import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from scipy.stats import norm
import scipy.stats as stats
from scipy.sparse import diags
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

from fisher_scoring import fs_probit
from metropolis_hastings import mh_probit

# Simulate data
X, y = make_classification(random_state=42, n_samples=1000, n_features=8, n_redundant=0, n_classes=2)
X = np.hstack((np.ones((X.shape[0], 1)), X))
y = y.reshape(-1, 1)

# Fisher Scoring
epsilon = 1e-6
b_fs = fs_probit(X, y, epsilon, verbose = False)
print()

# Metropolis-Hastings
#b_init = np.array([1 if i % 2 == 0 else -1 for i in range(X.shape[1])])
b_init = np.zeros(X.shape[1])

num_iter = 2000000
burn_iter = 200000
thinning = 10
prop_std = 0.001
prior_mean = 0
prior_std = 10

samples, b_mh, b_std = mh_probit(y, X, b_init, num_iter = num_iter, burn_iter = burn_iter, thinning = thinning, 
                            prop_std = prop_std, prior_mean = prior_mean, prior_std = prior_std, verbose = True)

# Plot
fig1, ax1 = plt.subplots(3, 3, figsize = (15, 15))
fig2, ax2 = plt.subplots(3, 3, figsize = (15, 15))

for i in range(X.shape[1]):
    sample = [sample[i] for sample in samples]
    #plt.plot(sample)
    ax1[i//3, i%3].plot(sample[burn_iter::thinning])
    ax1[i//3, i%3].set_title(f"beta_{i}")

    sm.graphics.tsa.plot_acf(sample[burn_iter::thinning], ax = ax2[i//3, i%3], title =  f"beta_{i}", lags=40)

fig1.suptitle("Beta plots")
fig1.tight_layout()
fig1.savefig("Beta_plot.png")


fig2.suptitle("Autocorrelations plots")
fig2.tight_layout()
fig2.savefig("ACL_plots.png")

# Compare two parameters
print("Fisher Scoring:")
print(b_fs)
print("Metropolis-Hastings:")
print(b_mh)
print("Differences (FS - MH):")
print(b_fs - b_mh)
