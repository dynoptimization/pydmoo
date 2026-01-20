"""
Includes modified code from [pymoo](https://github.com/anyoptimization/pymoo).

> Sources:
>
> - [dyn.py](https://github.com/anyoptimization/pymoo/blob/main/pymoo/problems/dyn.py)
>
> Licensed under the Apache License, Version 2.0. Original copyright and license terms are preserved.
"""

from abc import ABC
from math import ceil

from mpmath import mp
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem


class DynamicProblem(Problem, ABC):
    """Abstract base class for dynamic optimization problems."""
    pass


class DynamicApplProblem(DynamicProblem):
    """Dynamic optimization problem for real-world applications.

    This class defines dynamic optimization problems that model practical, real-world scenarios where the problem
    characteristics change systematically over time.

    Parameters
    ----------
    nt : int
        Severity of change. Controls how significantly the problem changes
        at each change point. Higher values indicate more substantial changes
        in problem characteristics.
    taut : int
        Frequency of change. Specifies how often (in generations) the problem
        undergoes changes. Lower values mean more frequent changes.
    t0 : int, optional
        The first change occurs after t0 generations, by default 50.
        That is, the generation at which a change occurs is (t0+1), (t0+taut+1), etc.
        This allows for an initial stabilization period before the first change.
    tau : int, optional
        Current simulation time counter (in generations), by default 1.
    time : float, optional
        Explicit simulation time value (overrides calculated time), by default None.
        Used for manual time control in specific scenarios.
    **kwargs : dict
        Additional keyword arguments passed to the parent Problem class.

    Attributes
    ----------
    tau : int
        Current simulation time counter in generations.
    nt : int
        Severity of change at each change point.
    taut : int
        Frequency of change between consecutive changes.
    t0 : int
        Initial stabilization period before first change occurs.

    Notes
    -----
    This class models real-world dynamic scenarios where:

    - Changes occur at predictable intervals (every `taut` generations)
    - Change severity is controlled by `nt` parameter
    - Initial period `t0` allows for system stabilization
    """

    def __init__(self, nt: int, taut: int, t0: int = 50, tau: int = 1, time: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau  # time counter
        self.nt = nt  # severity of change
        self.taut = taut  # frequency of change
        self.t0 = t0  # Initial time offset
        self._time = time

    def tic(self, elapsed: int = 1) -> None:

        # increase the time counter by one
        self.tau += elapsed

        # remove the cache of the problem to recreate ps and pf
        self.__dict__["cache"] = {}

    @property
    def time(self) -> float:
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
    def time(self, value: float) -> None:
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
    """Dynamic optimization problem for testing and benchmarking.

    Parameters
    ----------
    nt : int
        Severity of change. Controls how significantly the problem changes
        at each change point. Higher values indicate more substantial changes
        in problem characteristics.
    taut : int
        Frequency of change. Specifies how often (in generations) the problem
        undergoes changes. Lower values mean more frequent changes.
    t0 : int, optional
        The first change occurs after t0 generations, by default 50.
        That is, the generation at which a change occurs is (t0+1), (t0+taut+1), etc.
        This allows for an initial stabilization period before the first change.
    tau : int, optional
        Current simulation time counter (in generations), by default 1.
    time : float, optional
        Explicit simulation time value (overrides calculated time), by default None.
        Used for manual time control in specific scenarios.
    add_time_perturbation : bool, optional
        If True, adds perturbations to the time calculation, by default False.
    **kwargs : dict
        Additional keyword arguments passed to the parent Problem class.

    Attributes
    ----------
    tau : int
        Current simulation time counter in generations.
    nt : int
        Severity of change at each change point.
    taut : int
        Frequency of change between consecutive changes.
    t0 : int
        Initial stabilization period before first change occurs.
    add_time_perturbation : bool
        Flag indicating whether to add stochastic perturbations.

    Notes
    -----
    This class is designed for testing scenarios where:

    - Changes occur at predictable intervals (every `taut` generations)
    - Change severity is controlled by `nt` parameter
    - Initial period `t0` allows for system stabilization
    - Stochastic perturbations can be added for more complex testing
    - Reproducibility is important for benchmarking
    """

    def __init__(self, nt, taut, t0=50, tau=1, time=None, add_time_perturbation=False, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau  # time counter
        self.nt = nt  # severity of change
        self.taut = taut  # frequency of change
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
        r"""Time.

        Notes
        -----
        The discrete time $t$ is defined as follows:

        \begin{equation}
            t = \frac{1}{n_t} \left\lfloor \frac{\tau}{\tau_t} \right\rfloor + \frac{1}{n_t} \left(0.5 \times \frac{\pi_{\tau}}{9}\right), \ \tau = 0, 1, 2, \dots
        \end{equation}

        Here, $\pi_{\tau}$ is given by:

        \begin{equation}
            \pi_{\tau} =
            \begin{cases}
                0,                                                                                          & \text{if } \left\lfloor \frac{\tau}{\tau_t} \right\rfloor = 0, \\
                \text{the } \left\lfloor \frac{\tau}{\tau_t} \right\rfloor\text{-th decimal digit of } \pi, & \text{otherwise.}
            \end{cases}
        \end{equation}

        This formulation introduces a dynamic environment with an irregular change pattern. When $\pi_{\tau} = 0$, the time variation reduces to the commonly used form with a regular change pattern:

        \begin{equation} \label{eq:time_regular}
            t = \frac{1}{n_t} \left\lfloor \frac{\tau}{\tau_t} \right\rfloor, \ \tau = 0, 1, 2, \dots
        \end{equation}

        In the above expressions, $\tau$ denotes the generation counter, $n_t$ controls the severity of change, and $\tau_t$ represents the number of generations per time step.
        """
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
