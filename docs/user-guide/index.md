# API

## Example

```python
from pymoo.optimize import minimize

from pydmoo.algorithms.classic.nsga2_ae import NSGA2AE as DMOEA
from pydmoo.problems import DF1, GTS1
from pydmoo.problems.dyn import TimeSimulation


n_var = 10      # dimension of the decision variable
t0 = 100        # generations before the first environmental change
nc = 50         # total number of environmental changes
nt = 10         # severity of change
taut = 10       # frequency of change (generations between changes)
pop_size = 100  # population size

problem = GTS1(n_var=n_var, nt=nt, taut=taut, t0=t0)
algorithm = DMOEA(pop_size=pop_size)

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
