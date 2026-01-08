# pydmoo

pydmoo: Dynamic Multi-Objective Optimization in Python

---

## IMKT

```python
from pymoo.optimize import minimize

from pydmoo.algorithms import NSGA2IMKT as IMKT
from pydmoo.problems import GTS1
from pydmoo.problems.dyn import TimeSimulation

n_var = 10  # the dimension of decision variable
t0 = 100  # the first change occurs after t0 generations
nc = 50  # total number of changes
nt = 10  # severity of change
taut = 10  # frequency of change
pop_size = 100

problem = GTS1(n_var=n_var, nt=nt, taut=taut, t0=t0)
algorithm = IMKT(pop_size=pop_size)

seed = 2026
verbose = True

res = minimize(
    problem,
    algorithm,
    termination=("n_gen", taut * nc + t0),
    callback=TimeSimulation(),
    seed=seed,
    verbose=verbose,
)
```
