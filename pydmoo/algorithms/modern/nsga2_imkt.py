import numpy as np
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from pydmoo.algorithms.modern.nsga2_ktmm import NSGA2KTMM
from pydmoo.core.bounds import clip_and_randomize
from pydmoo.core.inverse import closed_form_solution
from pydmoo.core.sample_gaussian import univariate_gaussian_sample


class NSGA2IMKT(NSGA2KTMM):
    """Inverse Modeling with Knowledge Transfer.

    Inverse Modeling for Dynamic Multiobjective Optimization with Knowledge Transfer In objective Space.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 10
        self.denominator = 0.5

    def _response_mechanism(self):
        """Response mechanism."""
        """Inverse Modeling with Knowledge Transfer."""
        pop = self.pop
        X = pop.get("X")

        # recreate the current population without being evaluated
        pop = Population.new(X=X)

        # sample self.pop_size individuals in decision space
        samples_old = self.sampling_new_pop()

        # select self.pop_size/2 individuals with better convergence and diversity
        samples = samples_old[:int(len(samples_old)/2)]

        # knowledge in objective space
        means_stds, mean, std = self._in_decision_or_objective_space_1d(samples, "objective_space")
        mean_new, std_new = self._select_means_stds(means_stds, mean, std)

        # sample self.pop_size individuals in objective space
        F = univariate_gaussian_sample(mean_new, std_new, self.pop_size, random_state=self.random_state)

        # TODO
        # inverse mapping
        # X = FB
        B = closed_form_solution(samples.get("X"), samples.get("F"))

        # X = FB
        X = np.dot(F, B)

        # bounds
        if self.problem.has_bounds():
            xl, xu = self.problem.bounds()
            X = clip_and_randomize(X, xl, xu, random_state=self.random_state)

        # merge
        pop = Population.merge(samples_old, Population.new(X=X))

        return pop

    def sampling_new_pop(self):
        X = self.pop.get("X")

        if not self.problem.has_constraints():

            last_X = self.data.get("last_X", [])
            if len(last_X) == 0:
                last_X = X
            self.data["last_X"] = X

            d = np.mean(X - last_X, axis=0)

            radius = max(np.linalg.norm(d) / self.problem.n_obj, 0.1)

            X = X + d + self.random_state.uniform(low=-radius, high=radius, size=X.shape)

        # bounds
        if self.problem.has_bounds():
            xl, xu = self.problem.bounds()
            X = clip_and_randomize(X, xl, xu, random_state=self.random_state)

        samples = Population.new(X=X)
        samples = self.evaluator.eval(self.problem, samples)

        # do a survival to recreate rank and crowding of all individuals
        samples = RankAndCrowding().do(self.problem, samples, n_survive=len(samples))
        return samples


class NSGA2IMKT0(NSGA2IMKT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NSGA2IMKT1(NSGA2IMKT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 2
        self.denominator = 0.5


class NSGA2IMKT2(NSGA2IMKT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 4
        self.denominator = 0.5


class NSGA2IMKT3(NSGA2IMKT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 6
        self.denominator = 0.5


class NSGA2IMKT4(NSGA2IMKT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 8
        self.denominator = 0.5


class NSGA2IMKT5(NSGA2IMKT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 10
        self.denominator = 0.5


class NSGA2IMKT6(NSGA2IMKT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 12
        self.denominator = 0.5


class NSGA2IMKT7(NSGA2IMKT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 14
        self.denominator = 0.5


class NSGA2IMKT8(NSGA2IMKT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 16
        self.denominator = 0.5


class NSGA2IMKT9(NSGA2IMKT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 18
        self.denominator = 0.5


class NSGA2IMKT10(NSGA2IMKT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 20
        self.denominator = 0.5
