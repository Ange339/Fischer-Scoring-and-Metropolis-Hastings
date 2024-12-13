# Fischer-Scoring-and-Metropolis-Hastings
Here are experiments with two algorithms to infer Probit model parameters

## Probit Model
Probit Model is a Generalized Linear Model (GLM) used for binary data, in alternative to Logit model. The likelihood function is

```math
L(\beta; Y; X)= \prod_{i=1}^n \left[ \Phi(X_{i}^T\beta) \right]^{Y_i} \left[ 1 - \Phi(X_{i}^T\beta) \right]^{1-Y_i}
```
where $Y_i \overset{\text{iid}}{\sim} \text{Bern}(\mu_i)$, $\mu_i = \Phi(X_{i}^T\beta)$ where $\Phi$ is the CDF of a standard Gaussian - the last equality is also known as Link Function.


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
Z_{(t)} = X^T\beta_{(t)} + (Y - \mu) \left(\frac{\partial{X^T\beta_{(t)}}}{\partial{\mu_{(t)}}}\right) = X^T\beta_{(t)} + \frac{Y - \mu}{\phi(X^T\beta_{(t)})}
```

which is a (p x 1) vector with $\phi()$ denoting the standard Gaussian PDF.

The running time of this algorithm is $\mathcal{O}(k * (np^2 + p^3))$, where $k$ is the number of iterations, $n$ and $p$ are the shape of input matrix: $np^2 + p^3$ occurs because of matrix operation $X^TWX$ and its invertion.



