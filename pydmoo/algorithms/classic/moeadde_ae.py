import numpy as np
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from pydmoo.algorithms.base.dmoo.dmoeadde import DMOEADDE
from pydmoo.core.inverse import closed_form_solution


class MOEADDEAE(DMOEADDE):
    """Autoencoding.

    References
    ----------
    Feng, L., Zhou, W., Liu, W., Ong, Y.-S., and Tan, K. C. (2022).
    Solving dynamic multiobjective problem via autoencoding evolutionary search.
    IEEE Transactions on Cybernetics, 52(5), 2649–2662.
    https://doi.org/10.1109/TCYB.2020.3017017
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _response_mechanism(self):
        """Response mechanism."""
        pop = self.pop
        X = pop.get("X")

        # recreate the current population without being evaluated
        pop = Population.new(X=X)

        # predict via denoising autoencoding
        PSs = self.data.get("PSs", [])
        PSs.append(self.opt.get("X"))  # Parate Set
        PSs = PSs[-2:]
        self.data["PSs"] = PSs

        a = 0
        if len(PSs) == 2:
            # Pareto Set
            P, Q = PSs

            # Q = PM
            min_len = min(len(P), len(Q))
            M = closed_form_solution(Q[:min_len], P[:min_len])

            # X = QM
            X = np.dot(Q, M)

            # bounds
            if self.problem.has_bounds():
                xl, xu = self.problem.bounds()
                X = np.clip(X, xl, xu)  # not provided in the original reference literature

            # evalutate new population
            samples = self.evaluator.eval(self.problem, Population.new(X=X))
            a = min(int(self.pop_size / 2), len(samples))

            # do a survival to recreate rank and crowding of all individuals
            samples = RankAndCrowding().do(self.problem, samples, n_survive=a, random_state=self.random_state)

            pop[:a] = samples[:a]

        # randomly select solutions from previous Parate Set
        # This is to, first, preserve the high-quality solutions found along the evolutionary search process
        # second, to maintain the diversity of the population for further exploration of the evolutionary search.
        Q = self.opt.get("X")  # no-dominated solutions
        b = min(int(self.pop_size / 2), len(Q))
        idx = self.random_state.choice(np.arange(len(Q)), size=b)
        pop[a:(a + b)] = Population.new(X=Q[idx])

        # randomly generated solutions will be used to fill the population
        c = self.pop_size - a - b
        if c > 0:
            pop[(a + b):(a + b + c)] = self.initialization.sampling(self.problem, c, random_state=self.random_state)

        return pop
