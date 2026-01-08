import numpy as np


def closed_form_solution(X, Y):
    """
    Compute the least squares solution B = (YᵀY)⁻¹(YᵀX) for the linear system X ≈ YB.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        Target matrix containing dependent variables.
    Y : ndarray, shape (m, p)
        Design matrix containing independent variables.

    Returns
    -------
    B : ndarray, shape (p, n)
        Coefficient matrix that minimizes the Frobenius norm of (X - YB).

    Raises
    ------
    LinAlgError
        If YᵀY is singular and cannot be inverted, falls back to pseudo-inverse.

    Notes
    -----
    1. Solves the ordinary least squares problem in closed form.
    2. Automatically handles singular matrices by using pseudo-inverse when necessary.
    3. For numerical stability, consider using np.linalg.lstsq() in production code.

    """
    # Method 1
    # # Compute Y transpose multiplied by X: YᵀX (p x n)
    # YTX = np.dot(Y.T, X)

    # # Compute Y transpose multiplied by Y: YᵀY (p x p)
    # YTY = np.dot(Y.T, Y)

    # # Compute inverse of YᵀY with fallback to pseudo-inverse
    # try:
    #     YTY_inv = np.linalg.inv(YTY)
    # except np.linalg.LinAlgError:
    #     YTY_inv = np.linalg.pinv(YTY)

    # # Compute final solution: B = (YᵀY)⁻¹(YᵀX)
    # B = np.dot(YTY_inv, YTX)

    # Method 2
    B, residuals, rank, s = np.linalg.lstsq(Y, X, rcond=None)

    # Method 3
    # from scipy.linalg import lstsq
    # B = lstsq(Y, X, cond=None)[0]

    return B
