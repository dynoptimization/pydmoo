import numpy as np


def igp_based_predictor(PS, PF, F_, sigma_n_2):
    # algorithm 2: IGP-Based Predictor
    # algorithm 1: IGPR
    K_g = np.dot(PF, PF.T)
    K_g_noise = K_g + sigma_n_2 * np.eye(len(PF))
    k_inv = np.linalg.inv(K_g_noise)
    # k_inv = cholesky_inverse_numpy(K_g_noise)

    X_ = np.dot(np.dot(np.dot(F_, PF.T), k_inv), PS)
    return X_


def cholesky_inverse_numpy(A):
    """
    Compute the inverse of a symmetric positive definite (SPD) matrix using Cholesky decomposition.

    Parameters
    ----------
    A : numpy.ndarray
        A symmetric positive definite matrix (must satisfy A = A.T and all eigenvalues > 0)

    Returns
    -------
    numpy.ndarray
        The inverse matrix A^{-1} computed via Cholesky decomposition

    Raises
    ------
    AssertionError
        If input matrix is not symmetric or not positive definite
    """
    # 1. Check if matrix is symmetric and positive definite
    assert np.allclose(A, A.T), "Matrix must be symmetric"
    assert np.all(np.linalg.eigvals(A) > 0), "Matrix must be positive definite"

    # 2. Compute Cholesky decomposition A = L L^T
    # L is lower triangular matrix with positive diagonal entries
    L = np.linalg.cholesky(A)

    # 3. Compute inverse of L (triangular matrix inversion)
    # Since L is lower triangular, its inverse can be computed efficiently
    inv_L = np.linalg.inv(L)  # Alternative: solve triangular systems

    # 4. Compute inverse of A using A^{-1} = (L^{-1})^T L^{-1}
    A_inv = inv_L.T @ inv_L

    return A_inv
