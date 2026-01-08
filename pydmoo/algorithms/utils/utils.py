import numpy as np


def reconstruct_covariance_from_triu(triu_elements, n_features):
    """
    Reconstruct a symmetric covariance matrix from upper triangular elements.

    This function takes the flattened upper triangular elements (including diagonal)
    of a covariance matrix and reconstructs the full symmetric matrix by leveraging
    the symmetry property of covariance matrices.

    Parameters
    ----------
    triu_elements : array_like
        Flattened array containing the upper triangular elements of the covariance matrix,
        including the diagonal elements. The elements should be in row-major order.
        Shape: (n_triu_elements,) where n_triu_elements = n_features * (n_features + 1) // 2
    n_features : int
        Dimensionality of the feature space, which determines the size of the
        output covariance matrix (n_features x n_features)

    Returns
    -------
    cov_matrix : ndarray
        Reconstructed symmetric covariance matrix of shape (n_features, n_features).
        The matrix satisfies cov_matrix[i, j] == cov_matrix[j, i] for all i, j.

    Raises
    ------
    ValueError
        If the length of triu_elements does not match the expected number of
        upper triangular elements for the given n_features

    Notes
    -----
    The number of upper triangular elements (including diagonal) for an n x n matrix
    is given by: n_triu_elements = n * (n + 1) // 2

    Examples
    --------
    >>> triu_elements = np.array([1.0, 0.5, 0.3, 2.0, 0.4, 3.0])
    >>> n_features = 3
    >>> reconstruct_covariance_from_triu(triu_elements, n_features)
    array([[1. , 0.5, 0.3],
           [0.5, 2. , 0.4],
           [0.3, 0.4, 3. ]])
    """
    # Validate input dimensions
    expected_triu_elements = n_features * (n_features + 1) // 2
    if len(triu_elements) != expected_triu_elements:
        raise ValueError(
            f"Invalid number of triu_elements: expected {expected_triu_elements}, "
            f"got {len(triu_elements)}"
        )

    # Initialize output matrix with zeros
    # Shape: (n_features, n_features)
    cov_matrix = np.zeros((n_features, n_features), dtype=np.float64)

    # Get indices for upper triangular portion (including diagonal)
    # rows, cols: arrays of indices where row <= col
    rows, cols = np.triu_indices(n_features)

    # Fill upper triangular portion with input elements
    # This sets cov_matrix[i, j] for all i <= j
    cov_matrix[rows, cols] = triu_elements

    # Fill lower triangular portion using symmetry
    # Since covariance matrix is symmetric: cov_matrix[i, j] = cov_matrix[j, i]
    # This sets cov_matrix[j, i] = cov_matrix[i, j] for all i < j
    mask = rows != cols
    cov_matrix[cols[mask], rows[mask]] = cov_matrix[rows[mask], cols[mask]]

    return cov_matrix


def make_semidefinite(matrix, tol=1e-8):
    """
    Convert a symmetric matrix to positive semi-definite form.

    This function takes a symmetric matrix and ensures it is positive semi-definite
    by eigen-decomposition and thresholding of negative eigenvalues. The input matrix
    is first symmetrized, then any negative eigenvalues are set to zero.

    Parameters
    ----------
    matrix : array_like
        Input matrix to be converted to positive semi-definite form.
        Should be square and approximately symmetric.
        Shape: (n, n)
    tol : float, optional
        Tolerance for eigenvalue thresholding and symmetry check.
        Eigenvalues less than this value will be set to zero.
        Default: 1e-8

    Returns
    -------
    psd_matrix : ndarray
        Positive semi-definite matrix with the same shape as input.
        All eigenvalues are guaranteed to be >= 0.
        Shape: (n, n)

    Raises
    ------
    LinAlgError
        If the eigen-decomposition fails (e.g., matrix is not numerically symmetric)
    ValueError
        If input matrix is not square

    Notes
    -----
    The algorithm proceeds in three steps:
    1. Symmetrize the input matrix: (matrix + matrix.T) / 2
    2. Compute eigen-decomposition using eigh (optimized for symmetric matrices)
    3. Threshold negative eigenvalues to zero and reconstruct

    This method preserves the eigenvectors while modifying only the eigenvalues
    that violate positive semi-definiteness.

    Examples
    --------
    >>> import numpy as np
    >>> matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> make_semidefinite(matrix)
    array([[1. , 0.5],
           [0.5, 1. ]])

    >>> # Example with negative eigenvalues
    >>> matrix_indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    >>> print("Original eigenvalues:", np.linalg.eigvalsh(matrix_indefinite))
    Original eigenvalues: [-1.  3.]
    >>> psd_matrix = make_semidefinite(matrix_indefinite)
    >>> print("PSD eigenvalues:", np.linalg.eigvalsh(psd_matrix))
    PSD eigenvalues: [0. 3.]

    See Also
    --------
    numpy.linalg.eigh : Eigen decomposition for symmetric/Hermitian matrices
    numpy.diag : Create diagonal matrix from vector
    """
    # Input validation
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Input matrix must be square, got shape {matrix.shape}")

    # Step 1: Ensure matrix is symmetric
    # Average with transpose to eliminate numerical asymmetry
    # Result: symmetric_matrix = (matrix + matrix.T) / 2
    symmetric_matrix = (matrix + matrix.T) / 2

    # Step 2: Compute eigen-decomposition
    # eigh is preferred over eig for symmetric matrices (faster, more stable)
    # Returns:
    #   eigvals : 1D array of eigenvalues in ascending order
    #   eigvecs : 2D array where columns are corresponding eigenvectors
    eigvals, eigvecs = np.linalg.eigh(symmetric_matrix)

    # Step 3: Threshold negative eigenvalues to zero
    # Preserve eigenvalues >= tol, set others to 0
    # This ensures positive semi-definiteness while minimizing distortion
    eigvals_psd = np.where(eigvals < tol, 0.0, eigvals)

    # Step 4: Reconstruct matrix from modified eigenvalues
    # Matrix = V * diag(λ) * V.T where V contains eigenvectors as columns
    psd_matrix = eigvecs @ np.diag(eigvals_psd) @ eigvecs.T

    return psd_matrix
