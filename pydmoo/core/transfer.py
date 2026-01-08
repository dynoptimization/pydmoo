import numpy as np
from scipy.linalg import eigh


class TCA:
    """Transfer Component Analysis (TCA) for domain adaptation.

    TCA finds a latent space where source and target domain distributions are similar.

    Parameters
    ----------
    kernel_type : {'linear', 'rbf', 'poly'}, default='rbf'
        Type of kernel function to use:
        - 'linear': Linear kernel (dot product)
        - 'rbf': Radial Basis Function (Gaussian) kernel
        - 'poly': Polynomial kernel
    kernel_param : float, default=1.0
        Parameter for kernel function:
        - For 'rbf': gamma parameter (inverse of kernel width)
        - For 'poly': degree of polynomial
    dim : int, default=20
        Dimensionality of the latent space (number of components to keep)
    mu : float, default=0.5
        Regularization parameter for numerical stability

    Attributes
    ----------
    W : ndarray of shape (n_samples, dim) or None
        Learned transformation matrix. None before fitting.
    X_train : ndarray of shape (n_samples, n_features) or None
        Training data stored for transformation. None before fitting.
    """

    def __init__(self, kernel_type='rbf', kernel_param=1.0, dim=20, mu=0.5):
        # For the TCA parameters, we set the Gaussian kernel function to the default value and the expected dimensionality
        # was set to be 20. The value of μ was set to 0.5.
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param
        self.dim = dim
        self.mu = mu
        self.W = None
        self.X_train = None

    def fit(self, Xs, Xt):
        """Fit TCA model to source and target data.

        Parameters
        ----------
        Xs : ndarray of shape (ns_samples, n_features)
            Source domain feature matrix
        Xt : ndarray of shape (nt_samples, n_features)
            Target domain feature matrix

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Stack source and target data vertically
        X = np.vstack((Xs, Xt))
        n = X.shape[0]  # Total number of samples
        ns, nt = Xs.shape[0], Xt.shape[0]  # Number of source/target samples

        # Compute kernel matrix K using selected kernel function
        K = self._kernel(X, X)
        self.X_train = X  # Store for transform phase

        # Construct MMD (Maximum Mean Discrepancy) matrix
        # This matrix encodes the distribution difference between domains
        M = np.zeros((n, n))
        M[:ns, :ns] = 1/(ns*ns)  # Source-source block
        M[ns:, ns:] = 1/(nt*nt)  # Target-target block
        M[:ns, ns:] = M[ns:, :ns] = -1/(ns*nt)  # Cross blocks

        # Centering matrix H = I - 1/n * 11^T
        # Projects data onto the space orthogonal to the vector of ones
        H = np.eye(n) - np.ones((n, n))/n

        # Solve generalized eigenvalue problem:
        # (K(M)K + μI)^-1 K(H)K w = λw
        # Using pseudo-inverse for numerical stability
        A = np.linalg.pinv(K @ M @ K + self.mu*np.eye(n)) @ K @ H @ K

        # Compute eigenvalues and eigenvectors
        # eigh returns eigenvalues in ascending order
        eigvals, eigvecs = eigh(A)

        # Select top 'dim' eigenvectors corresponding to largest eigenvalues
        # [::-1] reverses order to get descending eigenvalues
        self.W = eigvecs[:, np.argsort(eigvals)[-self.dim:][::-1]]

        return self

    def transform(self, X):
        """Transform data using learned TCA components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, dim)
            Data projected to the latent space

        Raises
        ------
        ValueError
            If fit() hasn't been called before transform()
        """
        if self.W is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Compute kernel between new data and training data
        K = self._kernel(X, self.X_train)

        # Project to latent space: X' = K * W
        return K @ self.W

    def _kernel(self, X1, X2):
        """Compute kernel matrix between X1 and X2.

        Parameters
        ----------
        X1 : ndarray of shape (n_samples1, n_features)
            First set of samples
        X2 : ndarray of shape (n_samples2, n_features)
            Second set of samples

        Returns
        -------
        K : ndarray of shape (n_samples1, n_samples2)
            Kernel matrix

        Raises
        ------
        ValueError
            If kernel_type is not supported
        """
        if self.kernel_type == 'linear':
            return X1 @ X2.T  # Linear kernel: K(x,y) = x^T y
        elif self.kernel_type == 'rbf':
            # RBF kernel: K(x,y) = exp(-\gamma||x-y||^2)
            # Efficient computation using ||x-y||^2 = ||x||^2 + ||y||^2 - 2x^T y
            dist_sq = (np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T)
            return np.exp(-self.kernel_param * dist_sq)
        elif self.kernel_type == 'poly':
            # Polynomial kernel: K(x,y) = (x^T y + 1)^d
            return (X1 @ X2.T + 1) ** self.kernel_param
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}. Choose from 'linear', 'rbf', 'poly'")
