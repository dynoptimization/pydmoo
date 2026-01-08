import numpy as np


class ARModel:
    """
    Autoregressive (AR) model implementation from scratch.

    Parameters
    ----------
    p : int
        Order of the AR model (number of lagged observations)
    """

    def __init__(self, p):
        self.p = p  # AR order
        self.coef_ = None  # Model coefficients (including intercept if present)
        self.resid_ = None  # Residuals after fitting

    def fit(self, y, trend="c"):
        """
        Fit AR model to time series data.

        Parameters
        ----------
        y : ndarray, shape (M,)
            Time series data of length M
        trend : str, optional
            'c' for constant (default), 'n' for no intercept

        Returns
        -------
        self : returns an instance of self
        """
        M = len(y)
        if M <= self.p:
            raise ValueError(f"Time series length M={M} must be > order p={self.p}")

        # Construct design matrix X and target vector Y
        X = np.zeros((M - self.p, self.p))
        for i in range(self.p):
            X[:, i] = y[(self.p - i - 1): (M - i - 1)]

        Y = y[self.p:]

        # Add intercept if specified
        if trend == "c":
            X = np.column_stack([np.ones(X.shape[0]), X])

        # Solve least squares problem
        self.coef_ = np.linalg.lstsq(X, Y, rcond=None)[0]

        # Store residuals
        self.resid_ = Y - X @ self.coef_

        return self

    def predict(self, y, steps=1):
        """
        Predict future values using the fitted AR model.

        Parameters
        ----------
        y : ndarray, shape (K,)
            Input sequence (K >= p)
        steps : int, optional
            Number of steps to predict (default=1)

        Returns
        -------
        predictions : ndarray, shape (steps,)
            Predicted values
        """
        if len(y) < self.p:
            raise ValueError(f"Input length {len(y)} must be >= model order {self.p}")
        if self.coef_ is None:
            raise ValueError("Model must be fitted before prediction")

        predictions = np.zeros(steps)
        history = y[-self.p:].copy()  # Last p observations

        for i in range(steps):
            # Prepare input vector
            x = history[-self.p:][::-1]  # Latest p observations in reverse order

            # Add intercept if present (coef[0] is intercept)
            if len(self.coef_) == self.p + 1:
                x = np.insert(x, 0, 1)

            # Make prediction
            pred = np.dot(x, self.coef_)
            predictions[i] = pred

            # Update history
            history = np.append(history, pred)

        return predictions
