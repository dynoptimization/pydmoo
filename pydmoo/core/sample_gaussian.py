import numpy as np


def univariate_gaussian_sample(mean, std, n_samples=100, random_state=None):
    """
    Generate samples from a 1-dimensional Gaussian distribution.

    Parameters
    ----------
    mean : array_like, shape (n_features,)
        Mean values of the Gaussian distribution for each feature.
    std : array_like, shape (n_features,)
        Standard deviation of the Gaussian distribution for each feature.
    n_samples : int, optional
        Number of samples to generate (default=100).

    Returns
    -------
    ndarray, shape (n_samples, n_features)
        Random samples from the specified Gaussian distribution.

    Notes
    -----
    This function generates independent 1D Gaussian samples for each feature.
    """
    return random_state.normal(mean, std, size=(n_samples, len(mean)))  # np.random


def multivariate_gaussian_sample(mean, cov, n_samples=1, random_state=None):
    """
    Generate samples from a multivariate Gaussian distribution.

    Parameters
    ----------
    mean : array_like, shape (n_features,)
        Mean vector of the distribution.
    cov : array_like, shape (n_features, n_features)
        Covariance matrix of the distribution.
    n_samples : int, optional
        Number of samples to generate (default=1).

    Returns
    -------
    ndarray, shape (n_samples, n_features)
        Random samples from the multivariate Gaussian distribution.

    Raises
    ------
    ValueError
        If the covariance matrix is not positive-semidefinite.

    Notes
    -----
    Uses numpy.random.multivariate_normal for sampling.
    """
    return random_state.multivariate_normal(mean, cov, size=n_samples)  # np.random
