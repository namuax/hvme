# The causal engine

`humm.engine` is the reference for every number humm reports. It uses graph theory and exact probability calculations only, and it calls no language model. Gym labels, training rewards and validation verdicts all come from it.

| Module | What it does |
|---|---|
| `dsep` | d-separation, local Markov conditions, open paths |
| `identify` | identification in this order: generalised adjustment and the optimal (O-) set, front-door, instrumental variables; the ID algorithm for graphs with bidirected (confounded) pairs |
| `scm` | `BinarySCM` and `LinearSCM`: exact observational, interventional and counterfactual quantities; `BinarySCM.fit` estimates CPTs from data |
| `estimate` | evaluates the identified estimand on data (stratification for binary variables, linear regression for continuous ones) with bootstrap intervals |
| `refute` | placebo treatment, random common cause, subsample refutation |
| `discovery` | Markov equivalence class from (conditional) independence statements; checks a hypothesis in every graph of the class |

## Quantities

| Name | Rung | Meaning |
|---|---|---|
| `marginal` | 1 | P(Y = y) |
| `correlation` | 1 | P(Y = y \| X = x) − P(Y = y \| X = x′), optionally given other observed variables |
| `ate` | 2 | P(Y = y \| do(X = x)) − P(Y = y \| do(X = x′)) |
| `cde` | 2 | controlled direct effect with mediators fixed |
| `ett` | 3 | effect of treatment on the treated |
| `nde`, `nie` | 3 | natural direct and indirect effects |
| `pn`, `ps` | 3 | probability of necessity and of sufficiency (exact when identifiable, otherwise Tian–Pearl lower and upper bounds) |

Structural queries on the graph alone: `path` (parents and ancestors), `dsep`, `adjustment`, `frontdoor`, `instrument`, `identify`.

Hypotheses for causal discovery: `direct`, `via`, `cause`, `collider`, `confounder`. A hypothesis is implied only when it holds in every graph of the equivalence class.

## Example

```python
import networkx as nx
from humm.engine import BinarySCM, identify

g = nx.DiGraph([("season", "ad"), ("season", "sales"), ("ad", "traffic"), ("traffic", "sales"), ("ad", "sales")])
print(identify(g, "ad", "sales").strategy)          # backdoor, adjustment set {season}

scm = BinarySCM.random(g, seed=0)
print(scm.ate("ad", "sales"))                       # exact interventional effect
print(scm.nie("ad", "sales", mediators=["traffic"]))
```

See `notebooks/03_causal_engine.ipynb` for d-separation, the four identification cases, counterfactuals, estimates against the truth and refutation tests.
