import time

import numpy as np
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from pydmoo.algorithms.base.moo.moead import MOEAD


class DMOEAD(MOEAD):

    def __init__(self,
                 perc_detect_change=0.1,
                 eps=0.0,
                 **kwargs):

        super().__init__(**kwargs)
        self.perc_detect_change = perc_detect_change
        self.eps = eps

    def setup(self, problem, **kwargs):
        assert not problem.has_constraints(), f"{self.__class__.__name__} only works for unconstrained problems."
        return super().setup(problem, **kwargs)

    def _detect_change_sample_part_population(self):
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

    def _next_static_dynamic(self):
        # for dynamic environment
        pop = self.pop

        if self.state is None:

            change_detected = self._detect_change_sample_part_population()

            if change_detected:

                start_time = time.time()

                pop = self._response_change()

                # reevaluate because we know there was a change
                self.evaluator.eval(self.problem, pop)

                if len(pop) > self.pop_size:
                    # do a survival to recreate rank and crowding of all individuals
                    # Modified by DynOpt on Dec 21, 2025
                    # n_survive=len(pop) -> n_survive=self.pop_size
                    pop = RankAndCrowding().do(self.problem, pop, n_survive=self.pop_size, random_state=self.random_state)

                self.pop = pop

                self.data["response_duration"] = time.time() - start_time

        return pop

    def _response_change(self):
        pass


class DMOEADA(DMOEAD):

    def __init__(self,
                 perc_detect_change=0.1,
                 eps=0.0,
                 perc_diversity=0.3,
                 **kwargs):
        super().__init__(perc_detect_change=perc_detect_change,
                         eps=eps,
                         **kwargs)

        self.perc_diversity = perc_diversity

    def _response_change(self):
        pop = self.pop
        X = pop.get("X")

        # recreate the current population without being evaluated
        pop = Population.new(X=X)

        # find indices to be replaced (introduce diversity)
        I = np.where(self.random_state.random(len(pop)) < self.perc_diversity)[0]

        # replace with randomly sampled individuals
        pop[I] = self.initialization.sampling(self.problem, len(I), random_state=self.random_state)

        return pop


class DMOEADB(DMOEAD):

    def __init__(self,
                 perc_detect_change=0.1,
                 eps=0.0,
                 perc_diversity=0.3,
                 **kwargs):
        super().__init__(perc_detect_change=perc_detect_change,
                         eps=eps,
                         **kwargs)

        self.perc_diversity = perc_diversity

    def _response_change(self):
        pop = self.pop
        X = pop.get("X")

        # recreate the current population without being evaluated
        pop = Population.new(X=X)

        # find indices to be replaced (introduce diversity)
        I = np.where(self.random_state.random(len(pop)) < self.perc_diversity)[0]

        # replace by mutations of existing solutions (this occurs inplace)
        self.mating.mutation(self.problem, pop[I])

        return pop
