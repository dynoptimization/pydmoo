import numpy as np
from pymoo.core.population import Population

from pydmoo.algorithms.modern.nsga2_imkt import NSGA2IMKT
from pydmoo.core.bounds import clip_and_randomize
from pydmoo.core.distance import norm_mean_frobenius_distance
from pydmoo.core.inverse import closed_form_solution
from pydmoo.core.sample_gaussian import multivariate_gaussian_sample


class NSGA2IMKTN(NSGA2IMKT):
    """Inverse Modeling with Knowledge Transfer.

    Inverse Modeling for Dynamic Multiobjective Optimization with Knowledge Transfer In objective Space.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_pool = 10
        self.denominator = 0.5

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
        B = closed_form_solution(samples.get("X"), samples.get("F"))

        # X = FB
        X = np.dot(F, B)

        # Bounds
        if self.problem.has_bounds():
            xl, xu = self.problem.bounds()
            X = clip_and_randomize(X, xl, xu, random_state=self.random_state)

        # merge
        pop = Population.merge(samples_old, Population.new(X=X))

        return pop

    def _in_decision_or_objective_space_nd(self, samples, decision_or_objective="decision_space"):
        # decision space or objective space
        flag = "X" if decision_or_objective == "decision_space" else "F"

        means_covs = self.data.get("means_covs", [])

        flag_value = self.opt.get(flag)
        if len(flag_value) <= 1:
            flag_value = self.pop.get(flag)
            flag_value = flag_value[:2]

        m, c = np.mean(flag_value, axis=0), np.cov(flag_value.T)
        means_covs.append((m, 0.5 * (c.T + c), self.n_iter - 1))  # 1-based
        self.data["means_covs"] = means_covs

        flag_value = samples.get(flag)
        mean, cov = np.mean(flag_value, axis=0), np.cov(flag_value.T)
        return means_covs, mean, 0.5 * (cov.T + cov)

    def _select_means_covs(self, means_covs, mean_new, cov_new):
        # Unpack means and stds
        means = np.array([m[0] for m in means_covs])
        covs = np.array([m[1] for m in means_covs])

        # Calculate distances
        distances = np.array([
            norm_mean_frobenius_distance(mean, cov, mean_new, cov_new) for mean, cov in zip(means, covs)
        ])

        # Get top K closest
        top_k_idx = np.argsort(distances)[:self.size_pool]
        top_k_dist = distances[top_k_idx]
        top_k_means = means[top_k_idx]
        top_k_covs = covs[top_k_idx]

        # Update pool
        self._update_means_covs_pool(means_covs, top_k_idx)

        # Calculate weights
        weights = 1 / (top_k_dist + 1e-8)  # Add small epsilon to avoid division by zero
        weights = weights / (np.sum(weights) + self.denominator)

        # Weighted combination
        mean_new = (1 - np.sum(weights)) * mean_new + np.sum(weights[:, None] * top_k_means, axis=0)
        cov_new = (1 - np.sum(weights)) * cov_new + np.sum(weights[:, None, None] * top_k_covs, axis=0)

        # Symmetric matrix
        cov_new = 0.5 * (cov_new.T + cov_new)
        return mean_new, cov_new

    def _update_means_covs_pool(self, means_covs, top_k_idx) -> None:
        self.data["means_covs"] = [means_covs[i] for i in top_k_idx]
        return None


class NSGA2IMKTN0(NSGA2IMKTN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
