
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(66)

from rectangular_cr import drcr


def test_drcr():
    # Create samples with a random centered Gaussian distribution in 2D.
    num_samples = 1000
    sqrt_cov = np.random.randn(2, 2)
    white_samples = np.random.randn(num_samples, 2)
    samples = (sqrt_cov @ white_samples.T).T
    q = 0.95
    x_mode = np.zeros(2)
    # Define density:
    def unscaled_negative_log_density(x):
        w = np.linalg.solve(sqrt_cov.T, x)
        val = - 0.5 * np.sum(np.square(w))
        return val

    lb, ub, q_est = drcr(theta=q, samples=samples, g=unscaled_negative_log_density, mode=x_mode, verbose=True)
    samples_inside = [x for x in samples if np.all(x <= ub) and np.all(x >= lb)]
    num_inside = len(samples_inside)
    assert num_inside >= q * num_samples


def test_transformed_drcr():
    # Create samples with a random centered Gaussian distribution in 2D.
    num_samples = 1000
    sqrt_cov = np.random.randn(2, 2)
    white_samples = np.random.randn(num_samples, 2)
    samples = (sqrt_cov @ white_samples.T).T
    q = 0.95
    x_mode = np.zeros(2)

    # Define density:
    def unscaled_negative_log_density(x):
        w = np.linalg.solve(sqrt_cov.T, x)
        val = - 0.5 * np.sum(np.square(w))
        return val

    # Define filter: f(x) = 0.8*x1 + 0.2*x2, 0.2*x1 + 0.8*x2.
    ratio = 0.8
    transform_matrix = np.array([[ratio, 1 - ratio], [1 - ratio, ratio]])
    def transform(x):
        return transform_matrix @ x
    lb, ub, q_est = drcr(theta=q, samples=samples, transform=transform, g=unscaled_negative_log_density, mode=x_mode,
                                  verbose=True)
    transformed_samples = [transform(_) for _ in samples]
    transformed_samples_inside = [x for x in transformed_samples if np.all(x <= ub) and np.all(x >= lb)]
    num_inside = len(transformed_samples_inside)
    assert num_inside >= q * num_samples


def test_float_transform():
    """
    The program should also work when the transform outputs floats.
    """
    # Create samples with a random centered Gaussian distribution in 2D.
    num_samples = 1000
    sqrt_cov = np.random.randn(2, 2)
    white_samples = np.random.randn(num_samples, 2)
    samples = (sqrt_cov @ white_samples.T).T
    q = 0.95
    x_mode = np.zeros(2)

    # Define density:
    def unscaled_negative_log_density(x):
        w = np.linalg.solve(sqrt_cov.T, x)
        val = - 0.5 * np.sum(np.square(w))
        return val

    # Define a transform that outputs floats.
    def transform(x):
        return np.sum(x)

    lb, ub, q_est = drcr(theta=q, samples=samples, transform=transform, g=unscaled_negative_log_density, mode=x_mode,
                         verbose=True)
    transformed_samples = [transform(_) for _ in samples]
    transformed_samples_inside = [x for x in transformed_samples if np.all(x <= ub) and np.all(x >= lb)]
    num_inside = len(transformed_samples_inside)
    assert num_inside >= q * num_samples


