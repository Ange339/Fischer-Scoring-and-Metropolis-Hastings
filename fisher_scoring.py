import numpy as np
import pandas as pd
import time

from scipy.stats import norm
import scipy.stats as stats

from sklearn.datasets import make_classification


def probit(X, y, epsilon, verbose = True):
    n, p = np.shape(X)
    
    b_0 = np.zeros((p,1)) # Setting initial value of beta as 0
    Xb = np.dot(X,b_0) # Calculating Xb
    mu = norm.cdf(Xb) # Calculating mu
    W = np.diag(((norm.pdf(Xb)**2)/ (mu*(1-mu))).flatten()) # Calculating W
    Z = Xb + (y - mu)/norm.pdf(Xb) # Calculating Z

    iteration = 0 # Set the number of iterations

    start_time = time.process_time() # Start the timer
    while True: ### Algorithm Starts
        iteration += 1

        # Iterative updates
        b = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Z
        Xb = np.dot(X,b)
        mu = norm.cdf(Xb)
        W = np.diag(((norm.pdf(Xb)**2)/(mu*(1-mu))).flatten())
        Z = Xb + (y - mu)/norm.pdf(Xb)

        # Calculating the convergence criteria
        delta = np.linalg.norm(b-b_0)

        if verbose:
            print(f"Iteration:, {iteration}, Converging = {delta}")

        if delta < epsilon: # Checking for convergence
            print("\nConvergence reached with value:", delta)
            print("Number of iteration:", iteration)
            print(f"Time taken: {time.process_time() - start_time : .2f} seconds")
            break

        b_0 = b 
    
    print(f"\nParameters: {b.flatten()}")
    return(b.flatten())

