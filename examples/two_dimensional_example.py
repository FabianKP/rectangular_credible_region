
from math import exp, sqrt
import numpy as np
from scipy.optimize import lsq_linear

np.random.seed(123456)


def two_dimensional_example(num_samples: int):
    """
    Creates two-dimensional examples problem.
    """
    # Prior model.
    x_bar = np.array([0., 0.])
    sigma = 0.5

    # Forward model.
    x_true = np.array([1., 1.])
    fwd = np.array([[1., 4.], [8., 16.]])
    gamma = 1.
    y_bar = fwd @ x_true
    noise = gamma * np.random.randn(2)
    y = y_bar + noise

    # Compute MAP estimate by solving
    # min_x ||A x - b||_2^2
    # where A = [fwd / gamma, Id / sigma ], b = [y / gamma, x_bar / sigma].
    id_n = np.identity(2)
    a = np.concatenate([fwd / (sqrt(2) * gamma), id_n / (sqrt(2) * sigma)])
    b = np.concatenate([y / (sqrt(2) * gamma), x_bar / (sqrt(2) * sigma)])
    x_map = lsq_linear(A=a, b=b).x

    # Create posterior samples.
    w = np.random.randn(num_samples, b.size)
    samples = []
    for w_i in w:
        samples.append(lsq_linear(A=a, b=b + w_i).x)
    samples = np.array([samples]).reshape(num_samples, x_map.size)

    return x_map, x_true, a, b, samples
