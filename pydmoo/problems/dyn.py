"""
Includes modified code from [pymoo](https://github.com/anyoptimization/pymoo).

Sources:
    - [dyn.py](https://github.com/anyoptimization/pymoo/blob/main/pymoo/problems/dyn.py)

Licensed under the Apache License, Version 2.0. Original copyright and license terms are preserved.
"""

from abc import ABC
from math import ceil

from mpmath import mp
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem


class DynamicProblem(Problem, ABC):
    pass


class DynamicApplProblem(DynamicProblem):

    def __init__(self, nt, taut, t0=50, tau=1, time=None, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.nt = nt
        self.taut = taut
        self.t0 = t0
        self._time = time

    def tic(self, elapsed=1):

        # increase the time counter by one
        self.tau += elapsed

        # remove the cache of the problem to recreate ps and pf
        self.__dict__["cache"] = {}

    @property
    def time(self):
        if self._time is not None:
            return self._time
        else:
            # return 1 / self.nt * (self.tau // self.taut)

            # Calculate base time step
            delta_time = 1 / self.nt

            # Calculate time count considering initial offset
            count = max((self.tau + self.taut - (self.t0 + 1)), 0) // self.taut

            # Return time value
            return delta_time * count

    @time.setter
    def time(self, value):
        self._time = value

    def update_to_next_time(self):
        """Advance problem to the next significant time step.

        Returns
        -------
            elapsed: The actual time units advanced
        """
        # Calculate how many time steps to advance
        count = max((self.tau + self.taut - (self.t0 + 1)), 0) // self.taut

        # Calculate exact elapsed time needed to reach next discrete time point
        elapsed = int(count * self.taut + (self.t0 + 1) - self.tau)

        # Advance time by calculated amount
        self.tic(elapsed=elapsed)

        return elapsed


class DynamicTestProblem(DynamicProblem):

    def __init__(self, nt, taut, t0=50, tau=1, time=None, add_time_perturbation=False, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.nt = nt
        self.taut = taut
        self.t0 = t0  # Initial time offset - added by DynOpt Team
        self._time = time

        self.add_time_perturbation = add_time_perturbation  # Stochastic perturbation flag - added by DynOpt Team

    def tic(self, elapsed=1):

        # increase the time counter by one
        self.tau += elapsed

        # remove the cache of the problem to recreate ps and pf
        self.__dict__["cache"] = {}

    @property
    def time(self):
        if self._time is not None:
            return self._time
        else:
            # return 1 / self.nt * (self.tau // self.taut)

            # Modified by DynOpt Team
            # Calculate base time step
            delta_time = 1 / self.nt

            # Calculate time count considering initial offset
            count = max((self.tau + self.taut - (self.t0 + 1)), 0) // self.taut

            # Calculate perturbation ratio if enabled
            if not self.add_time_perturbation:
                ratio = 0

            else:
                # Use mathematical constants to generate deterministic perturbations
                mp.dps = max(ceil(10 + count), 10)
                mp_pi = 0 if count == 0 else int(str(mp.pi).split(".")[-1][count - 1])  # Extract digit from pi
                ratio = 0.5 * 1 / 9 * mp_pi

            # Return time value with optional perturbation
            return delta_time * count + delta_time * ratio

    @time.setter
    def time(self, value):
        self._time = value

    # Added by DynOpt Team
    def update_to_next_time(self):
        """Advance problem to the next significant time step.

        Returns
        -------
            elapsed: The actual time units advanced
        """
        # Calculate how many time steps to advance
        count = max((self.tau + self.taut - (self.t0 + 1)), 0) // self.taut

        # Calculate exact elapsed time needed to reach next discrete time point
        elapsed = int(count * self.taut + (self.t0 + 1) - self.tau)

        # Advance time by calculated amount
        self.tic(elapsed=elapsed)

        return elapsed


class TimeSimulation(Callback):
    """Callback for simulating time evolution in dynamic optimization problems.

    Handles time-linkage properties and time step updates.
    """

    def update(self, algorithm):
        """Update method called at each algorithm iteration."""
        problem = algorithm.problem

        # Added by DynOpt Team
        # Emulate time-linkage property: Update problem state based on current optimal solutions
        # Must execute before the problem.tic() to ensure proper time sequencing
        if hasattr(problem, "time_linkage") and hasattr(problem, "cal"):
            # Calculate time-linkage effects using current optimal objective values
            problem.cal(algorithm.opt.get("F"))

        # Advance time step for dynamic problem simulation
        if hasattr(problem, "tic"):
            problem.tic()  # Progress the dynamic problem to next time step
        else:
            raise Exception("TimeSimulation can only be used for dynamic test problems.")
