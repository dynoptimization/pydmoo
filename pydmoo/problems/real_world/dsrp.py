"""Provide the real-world applications of Dynamic Multi-Objective Optimization."""

import numpy as np

from pydmoo.problems.dyn import DynamicApplProblem


class DSRP(DynamicApplProblem):
    """Dynamic Speed Reducer Problem (DSRP).

    Kurpati, A., Azarm, S., and Wu, J. (2002).
    Constraint handling improvements for multiobjective genetic algorithms.
    Structural and Multidisciplinary Optimization, 23(3), 204–213.
    https://doi.org/10.1007/s00158-002-0178-2
    g9 = 1.9 - x5 + 1.1x7 <= 0 --- [x]
    g10 = f1 - 1300 <= 0 --------- [x]

    Zhang, Z., and Qian, S. (2011).
    Artificial immune system in dynamic environments solving time-varying non-linear constrained multi-objective problems.
    Soft Computing, 15(7), 1333–1349.
    https://doi.org/10.1007/s00500-010-0674-z
    g9 = 1.9 - x5 + 1.1x1 <= 0
    g10 = f1 - 4300 <= 0

    Zhang, Q., He, X., Yang, S., Dong, Y., Song, H., and Jiang, S. (2022).
    Solving dynamic multi-objective problems using polynomial fitting-based prediction algorithm.
    Information Sciences, 610, 868–886.
    https://doi.org/10.1016/j.ins.2022.08.020
    g9 = 1.9 - x5 + 1.1x1 <= 0
    g10 = f1 - 4300 <= 0

    Zou, J., Hou, Z., Jiang, S., Yang, S., Ruan, G., Xia, Y., and Liu, Y. (2025).
    Knowledge transfer with mixture model in dynamic multi-objective optimization.
    IEEE Transactions on Evolutionary Computation, 29(5), 1517–1530.
    https://doi.org/10.1109/TEVC.2025.3566481
    g9 = 1.9 - x5 + 1.1x1 <= 0
    g10 = f1 - 4300 <= 0
    """

    def __init__(self, n_var=7, n_obj=2, nt=10, taut=10, **kwargs):
        super().__init__(
            nt,
            taut,
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=11,
            xl=[2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0],
            xu=[3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5],
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        t = self.time
        x1, x2, x3, x4, x5, x6, x7 = x.T

        f1 = (
            a1_func(t) * x1 * x2**2 * (10 / 3 * x3**2 + a2_func(t) * x3 - a3_func(t))
            - a4_func(t) * x1 * (x6**2 + x7**2)
            + a5_func(t) * (x6**3 + x7**3)
            + a6_func(t) * (x4 * x6**2 + x5 * x7**2)
        )
        f2 = np.sqrt((745.0 * x4 / (x2 * x3)) ** 2 + 1.69e7) / (a7_func(t) * x6**3)

        out["F"] = np.column_stack([f1, f2])

        out["G"] = evaluate_constraints(x, f1)


def a1_func(t):
    a1 = 0.7854
    return a1 + t / 10


def a2_func(t):
    a2 = 14.933
    return a2 + t / 10


def a3_func(t):
    a3 = 43.0934
    return a3 + t / 10


def a4_func(t):
    a4 = 1.508
    return a4 + t / 10


def a5_func(t):
    a5 = 7.477
    return a5 + t / 10


def a6_func(t):
    a6 = 0.7854
    return a6 + t / 10


def a7_func(t):
    a7 = 0.1
    return a7 + (0.25 - 1 / (t + 4))


def evaluate_constraints(x, f1):
    """
    Evaluate all constraint functions for the speed reducer design problem.

    Parameters
    ----------
    x : numpy.ndarray of shape (n_samples, 7)
        Decision variable matrix where each row represents a solution vector
        [x1, x2, x3, x4, x5, x6, x7].
    f1 :
        Objective function value

    Returns
    -------
    g : numpy.ndarray of shape (n_samples, 11)
        Constraint violation matrix. g[:, i] <= 0 indicates the i-th constraint is satisfied.
        Each column corresponds to constraints g1 through g11.
    """
    n_samples = x.shape[0]
    g = np.zeros((n_samples, 11))

    # Extract design variables (0-based indexing in Python)
    x1 = x[:, 0]  # Gear face width
    x2 = x[:, 1]  # Teeth module
    x3 = x[:, 2]  # Number of teeth of pinion
    x4 = x[:, 3]  # Distance between bearings 1
    x5 = x[:, 4]  # Distance between bearings 2
    x6 = x[:, 5]  # Diameter of shaft 1
    x7 = x[:, 6]  # Diameter of shaft 2

    # g1: Geometric constraint related to gear dimensions
    g[:, 0] = 1.0 / (x1 * x2**2 * x3) - 1.0 / 27.0

    # g2: Second geometric constraint for gear design
    g[:, 1] = 1.0 / (x1 * x2**2 * x3**2) - 1.0 / 397.5

    # g3: Stiffness or stress constraint for shaft 1
    # Note: Original text has x5^5 but likely should be x5^3 for symmetry with g3
    g[:, 2] = x4**3 / (x2 * x3 * x6**4) - 1.0 / 1.93

    # g4: Stiffness or stress constraint for shaft 2
    g[:, 3] = x5**3 / (x2 * x3 * x7**4) - 1.0 / 1.93

    # g5: Constraint on gear size (teeth module x number of teeth)
    g[:, 4] = x2 * x3 - 40.0

    # g6: Upper bound constraint on face width to module ratio
    g[:, 5] = x1 / x2 - 12.0

    # g7: Lower bound constraint on face width to module ratio (rearranged form)
    g[:, 6] = 5.0 - x1 / x2

    # g8: Relationship between bearing distance 1 and shaft 1 diameter
    g[:, 7] = 1.9 - x4 + 1.5 * x6

    # g9: Relationship between bearing distance 2 and gear face width
    g[:, 8] = 1.9 - x5 + 1.1 * x1

    # g10: Upper bound constraint on weight objective function f1
    g[:, 9] = f1 - 4300.0

    # g11: Stress constraint for shaft 2
    g[:, 10] = np.sqrt((745.0 * x5 / (x2 * x3)) ** 2 + 1.515e8) / (0.1 * x7**3) - 1100.0

    return g
