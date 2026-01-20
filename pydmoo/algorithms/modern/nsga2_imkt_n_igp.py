from pymoo.core.population import Population

from pydmoo.algorithms.modern.nsga2_imkt_n import NSGA2IMKTN
from pydmoo.core.bounds import clip_and_randomize
from pydmoo.core.predictions import igp_based_predictor
from pydmoo.core.sample_gaussian import multivariate_gaussian_sample


class NSGA2IMKTNIGP(NSGA2IMKTN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 10
        self.denominator = 0.5

        self.delta_s = 0.01
        self.sigma_n = 0.01
        self.sigma_n_2 = self.sigma_n ** 2

    def _response_mechanism(self):
        """Response mechanism."""
        pop = self.pop
        X = pop.get("X")

        # recreate the current population without being evaluated
        pop = Population.new(X=X)

        # sample self.pop_size individuals in decision space
        samples_old = self.sampling_new_pop()

        # select self.pop_size/2 individuals with better convergence and diversity
        samples = samples_old[:int(len(samples_old)/2)]

        # knowledge in objective space
        means_stds, mean, cov = self._in_decision_or_objective_space_nd(samples, "objective_space")
        mean_new, cov_new = self._select_means_covs(means_stds, mean, cov)

        # sample self.pop_size individuals in objective space
        F = multivariate_gaussian_sample(mean_new, cov_new, self.pop_size, random_state=self.random_state)

        # TODO
        # inverse mapping
        # X = FB
        X = igp_based_predictor(samples.get("X"), samples.get("F"), F, self.sigma_n_2)

        # Bounds
        if self.problem.has_bounds():
            xl, xu = self.problem.bounds()
            X = clip_and_randomize(X, xl, xu, random_state=self.random_state)

        # merge
        pop = Population.merge(samples_old, Population.new(X=X))

        return pop


class NSGA2IMKTNIGP0(NSGA2IMKTNIGP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
