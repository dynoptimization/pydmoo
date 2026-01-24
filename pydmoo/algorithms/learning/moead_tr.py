import copy

import numpy as np
from pymoo.core.population import Population
from scipy.optimize import Bounds, minimize

from pydmoo.algorithms.base.dmoo.dmoead import DMOEAD
from pydmoo.core.transfer import TCA


class MOEADTr(DMOEAD):
    """Transfer learning (Tr).

    Transfer Learning-based Initial Population Generator (Tr-IPG)

    References
    ----------
    Jiang, M., Huang, Z., Qiu, L., Huang, W., and Yen, G. G. (2018).
    Transfer learning-based dynamic multiobjective optimization algorithms.
    IEEE Transactions on Evolutionary Computation, 22(4), 501–514.
    https://doi.org/10.1109/TEVC.2017.2771451
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ndim_ls = 20  # the dimension of latent space
        self.mu = 0.5

        #
        self._maxiter = max(self.pop_size, 100)  # default is 1000

    def _response_mechanism(self):
        """Response mechanism."""
        pop = self.pop
        X, F = pop.get("X", "F")

        last_time = self.data.get("last_time", 0)
        self.data["last_time"] = self.problem.time

        # source domain
        problem_ = copy.deepcopy(self.problem)
        problem_.time = last_time
        pop_s = self.initialization.sampling(problem_, self.pop_size, random_state=self.random_state)
        pop_s = self.evaluator.eval(problem_, pop_s)

        # target domain
        pop_t = self.initialization.sampling(self.problem, self.pop_size, random_state=self.random_state)
        pop_t = self.evaluator.eval(self.problem, pop_t)

        # Algorithm 1: TCA
        model = TCA(dim=self.ndim_ls, mu=self.mu)
        model.fit(pop_s.get("F"), pop_t.get("F"))

        # Remark3
        particles_latent_saace = model.transform(self.opt.get("F"))

        def dist_px(p, x, xl, xu):
            x = np.clip(x, xl, xu)
            pop_temp = Population.new(X=[x])
            pop_temp = self.evaluator.eval(self.problem, pop_temp)
            F = pop_temp.get("F")
            return np.sum((model.transform(F) - p) ** 2)

        X_ = []
        xl, xu = self.problem.bounds()
        for particle in particles_latent_saace:
            start = self.initialization.sampling(self.problem, 1, random_state=self.random_state).get("X")[0]
            start = np.clip(start, xl, xu)

            try:
                res = minimize(
                    lambda x: dist_px(particle, x, xl, xu),
                    start,
                    bounds=Bounds(xl, xu),
                    method="trust-constr",  # SLSQP, trust-constr, L-BFGS-B; In this paper, we use the interior point algorithm to solve the problem.
                    options={
                        "maxiter": self._maxiter,
                    },
                )
                x_opt = np.clip(res.x, xl, xu)
                X_.append(x_opt)

            except Exception as e:
                random_point = self.initialization.sampling(self.problem, 1, random_state=self.random_state).get("X")[0]
                X_.append(np.clip(random_point, xl, xu))

        # bounds
        X_ = np.array(X_)
        if self.problem.has_bounds():
            xl, xu = self.problem.bounds()
            X_ = np.clip(X_, xl, xu)  # not provided in the original reference literature

        # recreate the current population without being evaluated
        # merge
        pop = Population.merge(pop_t, Population.new(X=X_))

        return pop
