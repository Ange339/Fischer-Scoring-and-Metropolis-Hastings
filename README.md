# Fischer-Scoring-and-Metropolis-Hastings
Here are experiments with two algorithms to infer Probit model parameters

## Probit Model
Probit Model is a Generalized Linear Model (GLM) used for binary data, in alternative to Logit model. The likelihood function is

```math
L(\beta; y; X)= \prod_{i=1}^n \left[ \Phi(x_{i}^T\beta) \right]^{Y_i} \left[ 1 - \Phi(x_{i}^T\beta) \right]^{1-Y_i}
```
where $Y_i \overset{\text{iid}}{\sim} \text{Bern}(\mu_i)$, $\mu_i = \Phi(x_{i}^T\beta)$ where $\Phi$ is the CDF of a standard Gaussian - the last equality is also known as Link Function.


## Fisher Scoring
Fisher Scoring is a variation of Newton's method for the optimization, where the Hessian matrix is replaced with the Fisher Information Matrix.

In the context of GLM's parameters estimation, it can be shown that the iterative formula is:

```math
\beta_{(t+1)} = (X^T W_{(t)} X)^{-1} X^T W_{(t)} Z_{t}
```
where

```math
W_{(t)} = \frac{1}{Var(Y)} \left(\frac{\partial{X^T\beta_{(t)}}}{\partial{\mu_{(t)}}}\right)^{-2} = \frac{\left( \phi(X^T\beta_{(t)}) \right)^2}{\mu(1 -\mu)}
```

which is a (p x p) diagonal matrix that can be stored as sparsed matrix, and


```math
Z_{(t)} = X^T\beta_{(t)} + (y - \mu) \left(\frac{\partial{X^T\beta_{(t)}}}{\partial{\mu_{(t)}}}\right) = X^T\beta_{(t)} + \frac{y - \mu}{\phi(X^T\beta_{(t)})}
```

which is a (p x 1) vector with $\phi()$ denoting the standard Gaussian PDF.

The running time of this algorithm is $\mathcal{O}(k * (np^2 + p^3))$, where $k$ is the number of iterations, $n$ and $p$ are the shape of input matrix: $np^2 + p^3$ occurs because of matrix operation $X^TWX$ and its invertion.


## Metropolis-Hastings
Consider the Probit model under the Bayesian framework. Assuming a standard normal prior, the Bayesian Probit model is

```math
P(\beta|y,X) \propto \prod_{i=1}^n[\Phi(x_i^T\beta)]^{y_i}*[1-\Phi(x_i^T\beta)]^{1-y_i}*e^{-0.5(\beta^t\beta)}
```

which is intractable.

Metropolis-Hastings is a Monte Carlo Markov Chain (MCMC) method used for the Bayesian models, where the posterior is approximated by an iterative sampling technique. In each iteration, a proposal set of $\beta$ is sampled from a proposal distribution $Q(\beta|\beta_{t-1})$ conditional to $\beta_{(t-1)}$, and it is accepted with probability

```math
min\{1; \frac{P(\beta|y)Q(\beta_{(t-1)}|\beta)}{P(\beta_{(t-1)}|y)Q(\beta|\beta_{(t-1)})} \}
```

We'll use a normal distribution with $\beta_{(t-1)}$ as mean and a proposal $\sigma^2$ as the proposal distribution $Q$: $\sigma^2$ is a tuning parameter that control. We will further assume that $Q(\beta_{(t-1)}|\beta) = Q(\beta|\beta_{(t-1)})$, such that the acceptance probability is simply the minimum between 1 and the ratio of two posteriors.



