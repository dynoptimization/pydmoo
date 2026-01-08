import numpy as np


def uniform_sample_in_neighborhood(x, radius=0.1, bounds=None, size=1, random_state=None):
    """
    Generate uniformly distributed samples within hypercube neighborhoods of input points.

    Parameters
    ----------
    x : ndarray, shape (m, n)
        Input matrix of m solutions in n-dimensional space.
    radius : float or array_like, optional
        Neighborhood sampling radius. Can be:
        - Scalar: uniform radius for all dimensions (default=0.1)
        - Array of shape (n,): dimension-specific radii
        - Array of shape (m, n): solution-specific and dimension-specific radii
    bounds : tuple of array_like, optional
        Tuple containing (lower_bound, upper_bound) where each is of shape (n,).
        If provided, samples will be clipped to these bounds.
    size : int, optional
        Number of samples to generate per solution (default=1).

    Returns
    -------
    ndarray
        Sampled solutions:
        - If size=1: shape (m, n)
        - If size>1: shape (m, size, n)

    Notes
    -----
    1. Sampling is performed using uniform distribution within [-radius, +radius]
       around each solution coordinate.
    2. For bounded sampling, uses np.clip to enforce constraints.
    3. The function preserves input dtype for numerical precision.

    Examples
    --------
    >>> x = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> # Single sample with uniform radius
    >>> samples = uniform_sample_in_neighborhood(x, radius=0.5)
    >>> # Multiple samples with dimension-specific radii
    >>> samples = uniform_sample_in_neighborhood(x, radius=[0.1, 0.2], size=10)
    """
    # Input validation and shape processing
    x = np.asarray(x)
    m, n = x.shape  # m solutions, n dimensions

    # Process radius parameter
    if np.isscalar(radius):
        radius = np.full((1, n), radius)  # Broadcast scalar to all dimensions
    elif isinstance(radius, (list, np.ndarray)) and len(radius) == n:
        radius = np.reshape(radius, (1, n))  # Convert 1D vector to row vector

    # Generate random perturbations
    samples = x[:, np.newaxis, :] + random_state.uniform(low=-radius, high=radius, size=(m, size, n))  # np.random

    # Apply bounds constraint if provided
    if bounds is not None:
        lb, ub = np.asarray(bounds[0]), np.asarray(bounds[1])
        samples = np.clip(samples, lb, ub)

    return samples.squeeze()
