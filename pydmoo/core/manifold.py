import numpy as np


def manifold_prediction(X0, X1):
    """
    Predict the next manifold and compute its dispersion metric.

    Given two sequential populations in a manifold, this method:
    1. Centers the populations by removing their means
    2. Computes a dispersion metric (sigma) based on set distances

    Parameters
    ----------
    X0 : ndarray, shape (N, n)
        Population at time t-1, where:
        - N: number of points/samples
        - n: dimensionality of the manifold
    X1 : ndarray, shape (N, n)
        Population at time t

    Returns
    -------
    tuple (ndarray, float)
        - C1: ndarray, shape (N, n)
            Centered manifold at time t
        - variance: float
            Normalized dispersion metric computed as:
            variance = (D(C1,C0)^2) / n
            where D is the set distance between manifolds

    Notes
    -----
    1. The manifold is constructed by centering the input data
    2. The dispersion metric variance quantifies the normalized squared distance
       between consecutive manifolds
    3. Requires self.set_distance() method to be implemented
    4. Both input arrays must have same shape (N, n)

    """
    # Center the populations by removing column means
    C0 = X0 - np.mean(X0, axis=0)  # Centered manifold at t-1
    C1 = X1 - np.mean(X1, axis=0)  # Centered manifold at t

    return C1, set_distance(C1, C0)


def set_distance(A, B):
    """
    Compute the average minimum distance between two sets of points.

    The distance metric is defined as:
    D(A, B) = (1/|A|) ∑_{x∈A} min_{y∈B} ||x - y||

    Parameters
    ----------
    A : ndarray, shape (m, d)
        First set of points containing m samples in d-dimensional space.
    B : ndarray, shape (n, d)
        Second set of points containing n samples in d-dimensional space.

    Returns
    -------
    float
        The average minimum Euclidean distance between points in A and their
        nearest neighbors in B.

    Notes
    -----
    1. Uses Euclidean (L2) norm for distance computation.
    2. For empty sets, will raise ValueError.
    3. Computational complexity is O(m*n*d) where:
       - m = number of points in A
       - n = number of points in B
       - d = dimensionality

    Examples
    --------
    >>> A = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> B = np.array([[1.1, 2.1], [3.1, 4.1], [5.0, 6.0]])
    >>> set_distance(A, B)
    0.14142135623730953
    """
    # Compute pairwise Euclidean distances using broadcasting:
    # A[:, np.newaxis, :] reshapes to (m, 1, d)
    # B[np.newaxis, :, :] reshapes to (1, n, d)
    # Resulting subtraction produces (m, n, d) array
    distances = np.linalg.norm(A[:, np.newaxis, :] - B[np.newaxis, :, :], axis=2)

    # Find minimum distance for each point in A to any point in B
    min_distances = np.min(distances, axis=1)

    # Return average minimum distance
    return np.mean(min_distances)
