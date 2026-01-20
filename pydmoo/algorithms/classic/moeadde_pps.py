import numpy as np
from pymoo.core.population import Population

from pydmoo.algorithms.base.dmoo.dmoeadde import DMOEADDE
from pydmoo.core.ar_model import ARModel
from pydmoo.core.bounds import matrix_conditional_update
from pydmoo.core.manifold import manifold_prediction


class MOEADDEPPS(DMOEADDE):
    """Population Prediction Strategy (Center point prediction and manifold prediction).

    References
    ----------
    Zhou, A., Jin, Y., and Zhang, Q. (2014).
    A population prediction strategy for evolutionary dynamic multiobjective optimization.
    IEEE Transactions on Cybernetics, 44(1), 40–53.
    https://doi.org/10.1109/TCYB.2013.2245892
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.p = 3  # the order of the AR model
        self.M = 23  # the length of history mean point series

    def _response_mechanism(self):
        """Response mechanism."""
        pop = self.pop
        X = pop.get("X")

        # archive center points
        center_points = self.data.get("center_points", [])
        center_points.append(np.mean(self.opt.get("X"), axis=0))

        # the maximum length
        center_points = center_points[(-self.M):]
        self.data["center_points"] = center_points

        # archive populations
        Xs = self.data.get("Xs", [])
        Xs.append(self.pop.get("X"))  # pop
        Xs = Xs[-2:]
        self.data["Xs"] = Xs

        if len(center_points) >= (self.p + 1):

            C1, distance = manifold_prediction(Xs[0], Xs[1])
            n = C1.shape[1]  # Dimensionality of the manifold
            variance = (distance ** 2) / n

            center, variances = self.center_points_prediction(center_points)

            X = center + C1 + self.random_state.normal(loc=0, scale=np.sqrt(variances + variance), size=X.shape)

            # bounds
            if self.problem.has_bounds():
                xl, xu = self.problem.bounds()
                X = matrix_conditional_update(X, xl, xu, self.pop.get("X"))

            # recreate the current population without being evaluated
            pop = Population.new(X=X)

        else:

            # recreate the current population without being evaluated
            pop = Population.new(X=X)

            # randomly sample half of the population and reuse half from the previous search
            # when the history information is not enough to build an AR(p) model.

            # randomly sample half of the population
            a = int(self.pop_size / 2)
            pop[:a] = self.initialization.sampling(self.problem, a, random_state=self.random_state)

            # randomly reuse the other half from t Population
            Q = self.pop.get("X")
            b = self.pop_size - a
            idx = self.random_state.choice(np.arange(len(Q)), size=b)
            pop[a:] = Population.new(X=Q[idx])

        return pop

    def center_points_prediction(self, center_points):
        n = len(center_points[0])
        center = np.zeros(n)
        variances = np.zeros(n)
        for i in range(len(center)):
            data = [c[i] for c in center_points]
            model = ARModel(self.p).fit(data)
            predictions = model.predict(data, 1)
            center[i], variances[i] = predictions[0], np.mean(model.resid_ ** 2)
        return center, variances
