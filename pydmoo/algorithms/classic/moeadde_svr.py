import numpy as np
from pymoo.core.population import Population
from sklearn.svm import SVR

from pydmoo.algorithms.base.dmoo.dmoeadde import DMOEADDE


class MOEADDESVR(DMOEADDE):
    """Support Vector Regression (SVR).

    Notes
    -----
    [Official Python Code](https://github.com/LeileiCao/MOEA-D-SVR/blob/master/MOEAD-SVR%20.py)

    References
    ----------
    Cao, L., Xu, L., Goodman, E. D., Bao, C., and Zhu, S. (2020).
    Evolutionary dynamic multiobjective optimization assisted by a support vector regression predictor.
    IEEE Transactions on Evolutionary Computation, 24(2), 305–319.
    https://doi.org/10.1109/TEVC.2019.2925722
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # SVR
        self._q = 4  # the number of preceding values that are correlated with the target value (dimension of input samples in the SVR model)
        self._C = 1000  # the regularization constant in SVR model
        self._epsilon = 0.05  # the insensitive tube size in SVR model
        # self._gamma = 1/d  # the Gaussian RBF kernel parameter used in SVR model, and d is the number of variables

    def _response_mechanism(self):
        """Response mechanism."""
        pop = self.pop
        X = pop.get("X")

        old = self.data.get("stacked_X", None)
        if old is None:
            stacked_X = np.expand_dims(X, axis=0)
        else:
            stacked_X = np.concatenate((old, np.expand_dims(X, axis=0)), axis=0)
        self.data["stacked_X"] = stacked_X

        N, d = X.shape
        sol = np.zeros((N, d))
        K = len(stacked_X)

        if K < self._q + 2:
            # recreate the current population without being evaluated
            # Re-evaluate the current population, and update the reference point
            pop = Population.new(X=X)

            return pop

        # Precompute sliding window indices to avoid redundant calculations
        window_indices = np.lib.stride_tricks.sliding_window_view(np.arange(K), self._q + 1)

        for i in range(N):
            for j in range(d):
                # Extract the time series for this (i,j) position
                ts = stacked_X[:K, i, j]

                # Create training data using vectorized sliding windows
                train = ts[window_indices]
                x_train = train[:, :-1]
                y_train = train[:, -1]

                # Train SVR model (consider moving this outside loops if possible)
                # gamma if 'auto', uses 1 / n_features (not provided in code but provided in paper)
                # versionchanged:: 0.22
                # The default value of ``gamma`` changed from 'auto' to 'scale'.
                svr = SVR(kernel='rbf', epsilon=self._epsilon, C=self._C, gamma=1/d)
                model = svr.fit(x_train, y_train)

                # Make prediction
                sol[i, j] = model.predict(ts[-self._q:].reshape(1, -1))[0]

        # bounds
        if self.problem.has_bounds():
            xl, xu = self.problem.bounds()
            sol = np.clip(sol, xl, xu)  # provided in the original reference literature

        # recreate the current population without being evaluated
        pop = Population.new(X=sol)

        return pop
