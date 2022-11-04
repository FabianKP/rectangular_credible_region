"""
Created by Fabian on 04.11.2022.
"""

import numpy as np


def dfrcr(theta: float, samples: np.array, max_iter: int = 1000, verbose: bool = True):
    """
    Density-free version of DRCR.
    Uses the method described by Held (2004) to estimate the probability density.

    Parameters
    ----------
    theta : float
        The desired crediblity level. The function will return a box that contains at least theta * 100 % of
        the samples.
    samples : shape (n, d)
        The d-dimensional samples, stacked row-wise.
    max_iter: optional, int
        Maximum number of bisection steps.
    verbose : bool
        If True, the method will print information to the console.

    Returns
    -------
    lb : shape (d, )
        The lower bound of the simultaneous credible region.
    ub : shape (d, )
        The upper bound of the simultaneous credible region.
    theta_est : float
        The achieved credibility level, i.e. the number of samples inside the estimated interval divided by
        the overall number of samples.
    """
    # Check input
    if not np.isscalar(theta):
        raise TypeError("'theta' must be a scalar.")
    if not 0. < theta < 1.:
        raise ValueError("'theta' must be between 0 and 1.")
    if not samples.ndim == 2:
        raise ValueError("'samples' must be a numpy array of dimension 2.")
    n = samples.shape[0]
    # Pre-sort samples in ascending distance from the mode.
    raise NotImplementedError