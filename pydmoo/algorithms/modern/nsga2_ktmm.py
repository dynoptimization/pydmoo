import numpy as np
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from pydmoo.algorithms.base.dmoo.dnsga2 import DNSGA2
from pydmoo.core.sample_gaussian import univariate_gaussian_sample


class NSGA2KTMM(DNSGA2):
    """Knowledge Transfer with Mixture Model.

    Zou, J., Hou, Z., Jiang, S., Yang, S., Ruan, G., Xia, Y., and Liu, Y. (2025).
    Knowledge transfer with mixture model in dynamic multi-objective optimization.
    IEEE Transactions on Evolutionary Computation, in press.
    https://doi.org/10.1109/TEVC.2025.3566481
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.size_pool = 14  # the size of knowledge pool
        self.denominator = 0.5

    def _response_mechanism(self):
        """Response mechanism."""
        pop = self.pop
        X = pop.get("X")

        # recreate the current population without being evaluated
        pop = Population.new(X=X)

        # sample self.pop_size solutions in decision space
        samples_old = self.sampling_new_pop()

        # select self.pop_size/2 individuals with better convergence and diversity
        samples = samples_old[:int(len(samples_old)/2)]

        # knowledge in decision space
        means_stds_ps, mean, std = self._in_decision_or_objective_space_1d(samples, "decision_space")
        mean_new, std_new = self._select_means_stds(means_stds_ps, mean, std)

        # sample self.pop_size solutions in decision space
        X = univariate_gaussian_sample(mean_new, std_new, self.pop_size, random_state=self.random_state)

        # bounds
        if self.problem.has_bounds():
            xl, xu = self.problem.bounds()
            X = np.clip(X, xl, xu)  # not provided in the original reference literature

        # merge
        pop = Population.merge(samples_old, Population.new(X=X))

        return pop

    def _in_decision_or_objective_space_1d(self, samples, decision_or_objective="decision_space"):
        # decision space or objective space
        flag = "X" if decision_or_objective == "decision_space" else "F"

        means_stds = self.data.get("means_stds", [])

        flag_value = self.opt.get(flag)
        if len(flag_value) <= 1:
            flag_value = self.pop.get(flag)
            flag_value = flag_value[:2]

        means_stds.append((np.mean(flag_value, axis=0), np.std(flag_value, axis=0), self.n_iter - 1))  # 1-based
        self.data["means_stds"] = means_stds

        flag_value = samples.get(flag)
        mean, std = np.mean(flag_value, axis=0), np.std(flag_value, axis=0)
        return means_stds, mean, std

    def sampling_new_pop(self):
        samples = self.initialization.sampling(self.problem, self.pop_size)
        samples = self.evaluator.eval(self.problem, samples)

        # do a survival to recreate rank and crowding of all individuals
        samples = RankAndCrowding().do(self.problem, samples, n_survive=len(samples))
        return samples

    def _select_means_stds(self, means_stds, mean_new, std_new):
        # Unpack means and stds
        means = np.array([m[0] for m in means_stds])
        stds = np.array([m[1] for m in means_stds])

        # Calculate distances
        mean_diffs = means - mean_new
        std_diffs = stds - std_new

        distances = np.sqrt(np.sum(mean_diffs**2, axis=1) + np.sum(std_diffs**2, axis=1))

        # Get top K closest
        top_k_idx = np.argsort(distances)[:self.size_pool]
        top_k_dist = distances[top_k_idx]
        top_k_means = means[top_k_idx]
        top_k_stds = stds[top_k_idx]

        # Update pool
        self._update_means_stds_pool(means_stds, top_k_idx)

        # Calculate weights
        weights = 1 / (top_k_dist + 1e-8)  # Add small epsilon to avoid division by zero
        weights = weights / (np.sum(weights) + self.denominator)

        # Weighted combination
        mean_new = (1 - np.sum(weights)) * mean_new + np.sum(weights[:, None] * top_k_means, axis=0)
        std_new = (1 - np.sum(weights)) * std_new + np.sum(weights[:, None] * top_k_stds, axis=0)
        return mean_new, std_new

    def _update_means_stds_pool(self, means_stds, top_k_idx) -> None:
        self.data["means_stds"] = [means_stds[i] for i in top_k_idx]
        return None
