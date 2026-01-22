"""
Includes modified code from [pymoo](https://github.com/anyoptimization/pymoo).

Sources:
    - [dnsga2.py](https://github.com/anyoptimization/pymoo/blob/main/pymoo/algorithms/moo/dnsga2.py).

Licensed under the Apache License, Version 2.0. Original copyright and license terms are preserved.

The framework was adapted to enable efficient aggregation of change response strategies.
"""
import time

import numpy as np
from pymoo.core.population import Population

from pydmoo.algorithms.base.moo.nsga2 import NSGA2


class DNSGA2(NSGA2):
    """
    Dynamic Non-dominated Sorting Genetic Algorithm II (DNSGA2).

    Extension of NSGA2 for dynamic optimization problems.

    Parameters
    ----------
    perc_detect_change : float, default=0.1
        Percentage of population to sample for change detection (0 to 1).
    eps : float, default=0.0
        Threshold for change detection. Change is detected when mean squared
        difference exceeds this value.
    **kwargs
        Additional arguments passed to NSGA2 parent class.
    """

    def __init__(self,
                 perc_detect_change: float = 0.1,
                 eps: float = 0.0,
                 **kwargs):

        super().__init__(**kwargs)
        self.perc_detect_change = perc_detect_change
        self.eps = eps

    def setup(self, problem, **kwargs):
        assert not problem.has_constraints(), f"{self.__class__.__name__} only works for unconstrained problems."
        return super().setup(problem, **kwargs)

    def _detect_change_sample_part_population(self) -> bool:
        """
        Detect environmental changes by sampling part of the population.

        Returns
        -------
        change_detected : bool
            True if environmental change is detected, False otherwise.
        """
        pop = self.pop
        X, F = pop.get("X", "F")

        # the number of solutions to sample from the population to detect the change
        n_samples = int(np.ceil(len(pop) * self.perc_detect_change))

        # choose randomly some individuals of the current population to test if there was a change
        I = self.random_state.choice(np.arange(len(pop)), size=n_samples)
        samples = self.evaluator.eval(self.problem, Population.new(X=X[I]))

        # calculate the differences between the old and newly evaluated pop
        delta = ((samples.get("F") - F[I]) ** 2).mean()

        # if there is an average deviation bigger than eps -> we have a change detected
        change_detected = delta > self.eps
        return change_detected

    def _infill_static_dynamic(self) -> Population:
        """
        Perform infill with dynamic change detection and response.

        Returns
        -------
        Population
            Current population after potential response to environmental change.
        """
        # for dynamic environment
        pop = self.pop

        change_detected = self._detect_change_sample_part_population()

        if change_detected:

            start_time = time.time()

            pop = self._response_mechanism()

            # reevaluate because we know there was a change
            self.evaluator.eval(self.problem, pop)

            # do a survival to recreate rank and crowding of all individuals
            # Modified by DynOpt on Dec 21, 2025
            # n_survive=len(pop) -> n_survive=self.pop_size
            pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, random_state=self.random_state)

            self.pop = pop

            self.data["response_duration"] = time.time() - start_time

        return pop

    def _response_mechanism(self) -> Population:
        """
        Response mechanism for environmental change.

        Returns
        -------
        Population
            Population after applying response strategy.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError


class DNSGA2A(DNSGA2):
    """DNSGA2A.

    References
    ----------
    Deb, K., Rao N., U. B., and Karthik, S. (2007).
    Dynamic multi-objective optimization and decision-making using modified NSGA-II:
        A case study on hydro-thermal power scheduling.
    Evolutionary Multi-Criterion Optimization, 803–817.
    https://doi.org/10.1007/978-3-540-70928-2_60
    """

    def __init__(self,
                 perc_detect_change: float = 0.1,
                 eps: float = 0.0,
                 perc_diversity: float = 0.3,
                 **kwargs):
        super().__init__(perc_detect_change=perc_detect_change,
                         eps=eps,
                         **kwargs)

        self.perc_diversity = perc_diversity

    def _response_mechanism(self) -> Population:
        """Response mechanism."""
        pop = self.pop
        X = pop.get("X")

        # recreate the current population without being evaluated
        pop = Population.new(X=X)

        # find indices to be replaced (introduce diversity)
        I = np.where(self.random_state.random(len(pop)) < self.perc_diversity)[0]

        # replace with randomly sampled individuals
        pop[I] = self.initialization.sampling(self.problem, len(I), random_state=self.random_state)

        return pop


class DNSGA2B(DNSGA2):
    """DNSGA2B.

    References
    ----------
    Deb, K., Rao N., U. B., and Karthik, S. (2007).
    Dynamic multi-objective optimization and decision-making using modified NSGA-II:
        A case study on hydro-thermal power scheduling.
    Evolutionary Multi-Criterion Optimization, 803–817.
    https://doi.org/10.1007/978-3-540-70928-2_60
    """

    def __init__(self,
                 perc_detect_change: float = 0.1,
                 eps: float = 0.0,
                 perc_diversity: float = 0.3,
                 **kwargs):
        super().__init__(perc_detect_change=perc_detect_change,
                         eps=eps,
                         **kwargs)

        self.perc_diversity = perc_diversity

    def _response_mechanism(self) -> Population:
        """Response mechanism."""
        pop = self.pop
        X = pop.get("X")

        # recreate the current population without being evaluated
        pop = Population.new(X=X)

        # find indices to be replaced (introduce diversity)
        I = np.where(self.random_state.random(len(pop)) < self.perc_diversity)[0]

        # replace by mutations of existing solutions (this occurs inplace)
        self.mating.mutation(self.problem, pop[I])

        return pop
