# Fischer-Scoring-and-Metropolis-Hastings
Here are experiments with two algorithms to infer the Probit model parameters.

## Probit Model
The Probit model is a Generalized Linear Model (GLM) used for binary data, as an alternative to the Logit model. The likelihood function is:

```math
L(\beta; y; X)= \prod_{i=1}^n \left[ \Phi(x_{i}^T\beta) \right]^{Y_i} \left[ 1 - \Phi(x_{i}^T\beta) \right]^{1-Y_i}
```
where $Y_i \overset{\text{iid}}{\sim} \text{Bern}(\mu_i)$, and $\mu_i = \Phi(x_{i}^T\beta)$, where $\Phi$ is the CDF of a standard Gaussian. The last equality is also known as the Link Function


## Fisher Scoring
Fisher Scoring is a variation of Newton's method for the maximum likelihood optimization, where the Hessian matrix is replaced by the Fisher Information Matrix.

In the context of GLM parameter estimation, it can be shown that the iterative formula is:

```math
\beta_{(t+1)} = (X^T W_{(t)} X)^{-1} X^T W_{(t)} Z_{t}
```
where

```math
W_{(t)} = \frac{1}{Var(Y)} \left(\frac{\partial{X^T\beta_{(t)}}}{\partial{\mu_{(t)}}}\right)^{-2} = \frac{\left( \phi(X^T\beta_{(t)}) \right)^2}{\mu(1 -\mu)}
```

which is a (p x p) diagonal matrix that can be stored as a sparse matrix, and

```math
Z_{(t)} = X^T\beta_{(t)} + (y - \mu) \left(\frac{\partial{X^T\beta_{(t)}}}{\partial{\mu_{(t)}}}\right) = X^T\beta_{(t)} + \frac{y - \mu}{\phi(X^T\beta_{(t)})}
```

which is a (p x 1) vector, with $\phi()$ denoting the standard Gaussian PDF.

The running time of this algorithm is $\mathcal{O}(k * (np^2 + p^3))$, where $k$ is the number of iterations, and $n$ and $p$ are the dimensions of the input matrix. The term $np^2 + p^3$ occurs due to the matrix operation $X^T W X$ and its inversion.

## Metropolis-Hastings
Consider the Probit model within the Bayesian framework. Assuming a standard normal prior, the Bayesian Probit model is:

```math
P(\beta|y,X) \propto \prod_{i=1}^n[\Phi(x_i^T\beta)]^{y_i}*[1-\Phi(x_i^T\beta)]^{1-y_i}*e^{-0.5(\beta^t\beta)}
```

which is intractable.

Metropolis-Hastings is a Monte Carlo Markov Chain (MCMC) method used for Bayesian models, where the posterior is approximated by an iterative sampling technique. In each iteration, a proposed set of $\beta$ is sampled from a proposal distribution $Q(\beta|\beta_{t-1})$ conditional on $\beta_{(t-1)}$, and it is accepted with probability:

```math
min\{1; \frac{P(\beta|y)Q(\beta_{(t-1)}|\beta)}{P(\beta_{(t-1)}|y)Q(\beta|\beta_{(t-1)})} \}
```

We use a normal distribution with $\beta_{(t-1)}$ as the mean and $\sigma^2$ as the proposal distribution $Q$. The parameter $\sigma^2$ is a tuning parameter that controls the step size. We further assume that $Q(\beta_{(t-1)}|\beta) = Q(\beta|\beta_{(t-1)})$, so the acceptance probability is simply the minimum between 1 and the ratio of two posteriors.

Once the algorithm has iterated long enough and achieved stationarity, the estimate will be the Monte Carlo mean of the sampled parameters, considering burn-in and thinning.

The time complexity of this algorithm is $\mathcal{O}(k * np)$, where $k$ is the number of iterations, and $np$ is due to the matrix operation of $X^T\beta$.


