"""
Includes modified code from [pymoo](https://github.com/anyoptimization/pymoo).

Sources:

- [cec2015.py](https://github.com/anyoptimization/pymoo/blob/main/pymoo/problems/dynamic/cec2015.py)

Licensed under the Apache License, Version 2.0. Original copyright and license terms are preserved.

Add the method `_calc_pareto_front` for FDA4 and FDA5.
"""

from math import cos, fabs, floor, pi, sin, sqrt

import numpy as np

from pydmoo.problems.dyn import DynamicTestProblem
from pydmoo.problems.dynamic.df import get_PF


class DynamicCEC2015(DynamicTestProblem):

    def __init__(self, n_var=10, nt=10, taut=20, n_obj=2, xl=0.0, xu=1.0, vtype=float, **kwargs):
        super().__init__(nt, taut, n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=vtype, **kwargs)


class FDA2DEB(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import fda2_deb as f
        out["F"] = np.array([f(x, t) for x in X])


class FDA4(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, n_obj=3, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import FDA4 as f
        out["F"] = np.array([f(x, t) for x in X])

    # Added by DynOpt
    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        H = 20
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')

        f1 = np.cos(x1 * pi / 2) * np.cos(x2 * pi / 2)
        f2 = np.cos(x1 * pi / 2) * np.sin(x2 * pi / 2)
        f3 = np.sin(x1 * pi / 2)

        h = get_PF(np.array([f1, f2, f3]), False)
        return h


class FDA5(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, n_obj=3, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import FDA5 as f
        out["F"] = np.array([f(x, t) for x in X])

    # Added by DynOpt
    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        H = 20
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')

        G = fabs(sin(0.5 * pi * self.time))
        g = G
        F = 1 + 100 * pow(sin(0.5 * pi * self.time), 4)

        x1 = np.power(x1, F)
        x2 = np.power(x2, F)

        f1 = (1 + g) * np.cos(x1 * pi / 2) * np.cos(x2 * pi / 2)
        f2 = (1 + g) * np.cos(x1 * pi / 2) * np.sin(x2 * pi / 2)
        f3 = (1 + g) * np.sin(x1 * pi / 2)

        h = get_PF(np.array([f1, f2, f3]), True)
        return h


class DIMP2(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import DIMP2 as f
        out["F"] = np.array([f(x, t) for x in X])


class dMOP2(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import dMOP2 as f
        out["F"] = np.array([f(x, t) for x in X])


class dMOP3(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import dMOP3 as f
        out["F"] = np.array([f(x, t) for x in X])


class HE2(DynamicCEC2015):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import HE2 as f
        out["F"] = np.array([f(x, t) for x in X])


class HE7(DynamicCEC2015):

    def __init__(self, n_var=10, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import HE7 as f
        out["F"] = np.array([f(x, t) for x in X])


class HE9(DynamicCEC2015):

    def __init__(self, n_var=10, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        t = self.time
        from pymoo.vendor.gta import HE9 as f
        out["F"] = np.array([f(x, t) for x in X])
