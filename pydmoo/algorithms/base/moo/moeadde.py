import numpy as np
from pymoo.core.population import Population
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.operators.mutation.pm import PM

from pydmoo.algorithms.base.moo.moead import MOEAD


class MOEADDE(MOEAD):
    """MOEA/D-DE.

    Notes
    -----
    It is worth noting that there is a distinct modification in line 28 compared with the original framework of
    MOEA/D-DE. The newly generated solution competes with each member from the corresponding mating neighborhood
    (denoted as Pool in Algorithm 2). But in the original MOEA/D-DE framework, it only competes with two members from
    the corresponding mating neighborhood. This modification expands the replacement neighborhood to enhance the
    exploitation capability that is extremely important in dealing with DMOPs (Cao et al., 2020).

    References
    ----------
    Cao, L., Xu, L., Goodman, E. D., Bao, C., and Zhu, S. (2020).
    Evolutionary dynamic multiobjective optimization assisted by a support vector regression predictor.
    IEEE Transactions on Evolutionary Computation, 24(2), 305–319.
    https://doi.org/10.1109/TEVC.2019.2925722

    Hui Li and Qingfu Zhang. (2009).
    Multiobjective optimization problems with complicated pareto sets, MOEA/D and NSGA-II.
    IEEE Transactions on Evolutionary Computation, 13(2), 284–302.
    https://doi.org/10.1109/TEVC.2008.925798
    """
    def __init__(self, decomposition=Tchebicheff(), prob_neighbor_mating=0.8, **kwargs):
        super().__init__(decomposition=decomposition, prob_neighbor_mating=prob_neighbor_mating, **kwargs)

        # Neighborhood Selection
        # the number of neighbors considered during mating
        self.n_neighbors = 20
        self._delta = 0.8  # 0.9 (prob_neighbor_mating)

        # DE crossover
        self._cr = 0.5  # 1
        self._F_ = 0.5  # 0.5

        # Polynomial mutation
        self._eta = 20
        # self._pm = 1/d  # d is the number of variables
        self.mutation = PM(eta=self._eta)  # prob_var=1/d, default is min(0.5, 1/d)

    def _next(self):
        pop = self._next_static_dynamic()

        # iterate for each member of the population in random order
        for k in self.random_state.permutation(len(pop)):
            # Step 2.1 Selection of Mating/Update Range:
            # Select parents - use neighborhood with probability delta, else global selection
            pp = self.neighbors[k] if self.random_state.random() < self._delta else list(range(self.pop_size))

            # Step 2.2 Reproduction
            # Randomly select three distinct parents from the chosen pool
            a, b, c = self.random_state.choice(pp, size=3, replace=True)

            # Initialize problem parameters
            n_var = self.problem.n_var
            xl, xu = self.problem.xl, self.problem.xu

            X = pop.get("X")  # Position critical

            # mutation operator
            # Differential evolution mutation: v = x1 + F*(x2 - x3)
            mutation = X[a] + self._F_ * (X[b] - X[c])

            # crossover operator
            # Create mask for crossover operations (CR probability)
            mask = self.random_state.random(n_var) < self._cr

            # Combine mutation with target vector based on crossover probability
            V = np.where(mask, mutation, X[k])

            # mutation operator
            # polynomial mutation operator
            r = self.random_state.random(n_var)
            delta = np.where(
                self.random_state.random(n_var) < 0.5,
                np.power(2*r, 1/(1+self._eta)) - 1,  # First perturbation formula
                1 - np.power(2-2*r, 1/(1+self._eta))  # Alternative perturbation formula
            )
            perturb_mask = self.random_state.random(n_var) < (1/n_var)  # self._pm
            V[perturb_mask] += delta[perturb_mask] * (xu[perturb_mask] - xl[perturb_mask])

            # Step 2.3 Repair: If an element is out of the boundary, its value is reset to be a randomly selected value inside the boundary.
            # Ensure solution stays within bounds
            V = np.clip(V, xl, xu)

            # Individual
            off = Population.new(X=np.array([V]))[0]

            # evaluate the offspring
            off = yield off

            # Step 2.4 Update
            # update the ideal point
            self.ideal = np.min(np.vstack([self.ideal, off.F]), axis=0)

            # Step 2.5 Update of Solutions
            # It is worth noting that there is a distinct modification in line 28 compared with the original framework of MOEA/D-DE.
            # The newly generated solution competes with each member from the corresponding mating neighborhood (denoted as Pool in Algorithm 2).
            # But in the original MOEA/D-DE framework, it only competes with two members from the corresponding mating neighborhood.
            # This modification expands the replacement neighborhood to enhance the exploitation capability that is extremely important in dealing with DMOPs.
            # now actually do the replacement of the individual is better
            self._replace(k, off)
