DRCR - Density-guided rectangular credible regions for multivariate probability distributions
===

This Python module contains a method to estimate 
*rectangular credible regions* (sometimes also called
*simultaneous credible bands* or *simultaneous credible intervals*) given samples from 
a unimodal, multivariate probability distribution.

Given a unimodal probability density function $p(x)$ on $\mathbb{R}^d$ and
samples $x^{(1)}, \ldots, x^{(N)}$, the method computes a small, $d$-dimensional
box that contains a given percentage (e. g. 95% or 99%) of the samples.

Since the method requires access to the probability density function (or something that is
proportional to it), we call it **DRCR** for 
"**D**ensity-guided **R**ectangular **C**redible **R**egion".

For a more detailed explanation, see [this notebook]().


Installation
---


Usage
---

```python

# Assuming 'samples' is a two-dimensional array of samples, 'neg_log_dens' is a callable
# negative log-density function with mode 'x_mode'.

from rectangular_cr import drcr

# Want 95%-credible region.
theta = 0.95

lb, ub, theta_est = drcr(theta, samples, neg_log_dens, x_mode)
# The rectangular credible region is now given by {x : lb <= x <= ub},
# and it contains theta_est*100 % of the samples.
```


