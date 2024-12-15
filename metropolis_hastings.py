import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from scipy.stats import norm
import scipy.stats as stats
from scipy.sparse import diags
import statsmodels.api as sm


def log_likelihood(y, X, b):
    Xb = np.dot(X, b)
    mu = stats.norm.cdf(Xb)
    mu = np.clip(mu, 1e-10, 1 - 1e-10) # Avoiding log(0)
    log_lik = np.sum(np.log(mu)*y + np.log(1 - mu)*(1 - y))
    return log_lik

def log_prior(b, prior_mean, prior_std):
    prior = stats.norm.logpdf(b, loc=prior_mean, scale=prior_std)
    return np.sum(prior)

def mh_probit(y, X, b_0, num_iter = 1000, burn_iter = 200, thinning = 2,
                prop_std = 1, prior_mean = 0,  prior_std = 1, verbose = True):
    
    b_curr = b_0
    samples = []
    
    total_accepted = 0

    start_time = time.process_time() # Start the timer
    for i in range(num_iter):
        # Propose new beta
        b_prop = np.random.normal(b_curr, prop_std, size = len(b_curr))
        
        # Proposal posterior
        prop_log_lik = log_likelihood(y, X, b_prop)
        prop_log_prior = log_prior(b_prop, prior_mean, prior_std)
        prop_log_posterior = prop_log_lik + prop_log_prior

        # Current posterior
        curr_log_lik = log_likelihood(y, X, b_curr)
        curr_log_prior = log_prior(b_curr, prior_mean, prior_std)
        curr_log_posterior = curr_log_lik + curr_log_prior

        # Acceptance ratio
        log_acceptance_ratio = prop_log_posterior - curr_log_posterior
        log_acceptance_ratio = np.clip(log_acceptance_ratio, -700, 700) # Avoiding overflow
        acceptance_ratio = np.exp(log_acceptance_ratio)
        acceptance_prob = min(1, acceptance_ratio)

        ## Acceptance
        if acceptance_prob > np.random.rand():
            b_curr = b_prop
            accepted = True
        else:
            accepted = False
        total_accepted += accepted

        ## Save sample
        samples.append(b_curr)
        
        ## Print
        if verbose and i % 100 == 0:
            print(f"Iteration: {i}, Accepted: {total_accepted}, Acceptance rate: {total_accepted/(i+1)*100:.2f}")
    
    samples = np.array(samples)
    b_mean = np.mean(samples[burn_iter::thinning], axis = 0)
    b_std = np.std(samples[burn_iter::thinning], axis = 0)

    print(f"\nTotal accepted: {total_accepted}")
    print(f"Total Acceptance rate: {total_accepted/num_iter*100 :.2f}")
    print(f"Time taken: {time.process_time() - start_time : .2f} seconds")
    print(f"Parameters: {b_mean}")
    print(f"Standard Deviation: {b_std}")

    return samples, b_mean, b_std
