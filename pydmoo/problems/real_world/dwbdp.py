import numpy as np

from pydmoo.problems.dyn import DynamicApplProblem


class DWBDP(DynamicApplProblem):
    r"""Dynamic Welded Beam Design Problem (DWBDP).

    A multi-objective optimization problem with time-varying constraints and objectives
    that minimizes both manufacturing cost and beam deflection under dynamic loading conditions.

    Objectives
    ----------
    f1 : float
        Manufacturing cost ($), comprising setup cost, welding labor cost, and material cost
    f2 : float
        Beam deflection (inch) under applied load

    Variables
    ---------
    x : ndarray, shape (4,)
        x[0] : weld thickness h (inch)
        x[1] : weld length l (inch)
        x[2] : beam height t (inch)
        x[3] : beam thickness b (inch)

    Constraints
    -----------
    g1 : float
        Shear stress constraint ($\tau <= 13,600$)
    g2 : float
        Bending stress constraint ($\sigma ≤ 30,000$)
    g3 : float
        Geometric constraint ($h <= b$)
    g4 : float
        Buckling constraint ($P <= P_critical$)

    Notes
    -----
    It should be noted that in the formulation [1], the variable used in the expression
    for the resultant shear stress $\tau$ is $x_2$, rather than $x_3$ as may appear in certain
    alternative formulations. Readers are advised to verify the specific variable
    definitions and indexing used in the respective reference to ensure consistency
    in implementation.

    The problem incorporates temporal variations through time-dependent loading
    conditions P(t), making it suitable for testing dynamic multi-objective
    optimization algorithms.

    References
    ----------
    Oyama, A., Shimoyama, K., and Fujii, K. (2005).
    New constraint-handling method for multi-objective multi-constraint evolutionary optimization and its application to space plane design.
    Evolutionary and Deterministic Methods for Design, Optimization and Control with Applications to Industrial and Societal Problems.
    https://ladse.eng.isas.jaxa.jp/papers/eurogen2005.pdf

    Zhang, Z., and Qian, S. (2011).
    Artificial immune system in dynamic environments solving time-varying non-linear constrained multi-objective problems.
    Soft Computing, 15(7), 1333–1349.
    https://doi.org/10.1007/s00500-010-0674-z

    Zhang, Q., He, X., Yang, S., Dong, Y., Song, H., and Jiang, S. (2022).
    Solving dynamic multi-objective problems using polynomial fitting-based prediction algorithm.
    Information Sciences, 610, 868–886.
    https://doi.org/10.1016/j.ins.2022.08.020

    https://www.mathworks.com/help/gads/multiobjective-optimization-welded-beam.html

    https://github.com/Amir-M-Vahedi/Welded-Beam-Design-Optimization/blob/main/Welded_Beam_Design_Project_D3.py
    """

    def __init__(self, n_var=4, n_obj=2, nt=1, taut=10, **kwargs):
        # COMPULSORY
        nt = 1

        super().__init__(
            nt, taut, n_var=n_var, n_obj=n_obj, n_ieq_constr=4, xl=[55, 75, 1000, 2], xu=[80, 110, 3000, 20], **kwargs
        )

        # Material properties and constants
        self.E = 3.0e7  # Young's modulus (psi)
        self.G = 1.2e7  # Shear modulus (psi)
        self.L = 14.0  # Beam length (inch)

        # Load cases: (time, load in lb)
        self.load_cases = {0: 10000, 1: 80000, 2: 6000, 3: 3000}

        # [55, 80],      # x1: weld thickness bounds
        # [75, 110],     # x2: weld length bounds
        # [1000, 3000],  # x3: beam height bounds
        # [2, 20],       # x4: beam thickness bounds

        #
        # lb = [55, 75, 1000, 2]
        # ub = [80, 110, 3000, 20]

        #
        # lb = [0.125, 0.1, 0.1, 0.125]
        # ub = [5, 10, 10 ,5]

    def _evaluate(self, x, out, *args, **kwargs):
        t = self.time
        x1, x2, x3, x4 = x.T
        P = self.load_cases[int(t % 4)]

        f1 = f1 = 1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14 + x2)

        f2 = (4 * P * self.L**3) / (self.E * x3**3 * x4)

        out["F"] = np.column_stack([f1, f2])

        out["G"] = self.evaluate_constraints(x, P)

    def calculate_auxiliary_variables(self, x, P) -> tuple:
        """Calculate intermediate variables needed for constraint evaluation.

        Parameters
        ----------
        x : np.ndarray, shape (4,)
            Design variables [x1, x2, x3, x4]
        P : float
            Applied load (lb)

        Returns
        -------
        tuple : (tau, pc)
            tau : float - Resultant shear stress (psi)
            pc : float - Critical buckling load (lb)
        """
        x1, x2, x3, x4 = x.T

        # Radius to neutral axis
        R = np.sqrt(x2**2 / 4 + ((x1 + x3) / 2) ** 2)

        # Polar moment of inertia
        J = 2 * np.sqrt(2) * x1 * x2 * (x2**2 / 12 + ((x1 + x3) / 2) ** 2)

        # Bending moment
        M = P * (self.L + x2 / 2)

        # Primary and secondary shear stresses
        tau1 = P / (np.sqrt(2) * x1 * x2)
        tau2 = (M * R) / J

        # Resultant shear stress
        tau = np.sqrt(tau1**2 + 2 * tau1 * tau2 * (x2 / (2 * R)) + tau2**2)

        # Critical buckling load
        pc_numerator = 4.013 * self.E * x3 * x4**3
        pc_denominator = 6 * self.L**2
        pc_adjustment = 1 - (x3 / (2 * self.L)) * np.sqrt(self.E / (4 * self.G))
        pc = (pc_numerator / pc_denominator) * pc_adjustment

        return tau, pc

    def evaluate_constraints(self, x, P):
        """Evaluate all constraint functions.

        Parameters
        ----------
        x : np.ndarray, shape (4,)
            Design variables [x1, x2, x3, x4]

        Returns
        -------
        np.ndarray
            Constraint values (should be <= 0 for feasibility)
        """
        n_samples = x.shape[0]
        g = np.zeros((n_samples, 4))

        x1, _, x3, x4 = x.T

        # Evaluate constraints for specific load case
        tau, pc = self.calculate_auxiliary_variables(x, P)

        # g1: Shear stress constraint (tau <= 13,600 psi)
        g[:, 0] = tau - 13600

        # g2: Bending stress constraint (sigma <= 30,000 psi)
        g[:, 1] = (6 * P * self.L) / (x4 * x3**2) - 30000

        # g3: Geometric constraint (x1 <= x4)
        g[:, 2] = x1 - x4

        # g4: Buckling constraint (P <= pc)
        g[:, 3] = P - pc

        return g
