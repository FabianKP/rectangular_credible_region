"""
Created by Fabian on 30.10.2022.
"""

import numpy as np
from scipy.stats import rankdata
from typing import Tuple


def besag_method(theta: float, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes a rectangular simultaneous credible region using the method proposed by
    Besag, Green, Higdon and Mengersen (1995).
    That is, it computes a multidimensional box [lb, ub] that contains at least theta * 100% of the samples.

    Parameters
    ----------
    theta : float
        The desired credibility level. The function will return a box that contains at least theta * 100 % of
        the samples.
    samples : shape (n, d)
        The d-dimensional samples, stacked row-wise.

    Returns
    -------
    lb : shape (d, )
        The lower bound of the simultaneous credible region.
    ub : shape (d, )
        The upper bounds of the simultaneous credible region.
    theta_est : float
        The achieved credibility level, i.e. the number of samples inside the estimated interval divided by
        the overall number of samples.
    """
    # Check the input.
    assert samples.ndim == 2, "'samples' must be a 2-dimensional array."
    assert np.isscalar(theta), "'alpha' must be a scalar."
    assert 0. < theta < 1., "'alpha' must be in the interval (0, 1)."
    n, d = samples.shape
    # Let k = ceil((1 - alpha) * n).
    k = np.ceil(theta * n).astype(int)
    # Order the samples separately for each component.
    # Let r be the array of ranks, i.e. r[t, i] corresponds to the t-th largest value of the i-th component.
    r = rankdata(samples, axis=0, method="ordinal") - 1     # I want ranks to begin at 0.
    # Store the ordered samples in a vector x_ord.
    x_ord = np.sort(samples, axis=0)
    # Let s be the sequence {max(max_i r[t, i], n - min_i r[t, i]), t = 1,..., n}.
    s1 = np.max(r, axis=1)
    s2 = n - 1 - np.min(r, axis=1)
    s_pre = np.row_stack([s1, s2])
    s = np.max(s_pre, axis=0)
    # Let t_star be the k-th order statistic from the s. Then, in most cases, t_star is the smallest integer such that
    # x_ord[n - 1 - t_star, i] <= samples[t, i] <= x_ord[t_star, i] for all i and at least k values of t.
    s_ord = np.sort(s)
    t_star = s_ord[k - 1]
    # With these definitions, lb[i] = x_ord[n - 1 - t_star, i], ub = x_ord[t_star, i] define the
    # lower and upper bound for the i-th component of the simultaneous credible interval.
    lb = x_ord[n - 1 - t_star]
    ub = x_ord[t_star]

    # Compute theta_est.
    num_inside = len([x for x in samples if np.all(x >= lb) and np.all(x <= ub)])
    theta_est = num_inside / samples.shape[0]

    return lb, ub, theta_est


def _check_t_star(k, t_star, x_ord, samples):
    n = samples.shape[0]
    lb = x_ord[n - 1 - t_star]
    ub = x_ord[t_star]
    samples_inside = [x for x in samples if np.all(x >= lb) and np.all(x <= ub)]
    num_inside = len(samples_inside)
    return (num_inside >= k)