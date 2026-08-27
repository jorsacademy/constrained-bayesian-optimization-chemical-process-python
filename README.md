# Constrained Bayesian Optimization of a Parallel-Reaction CSTR

A physically motivated industrial-process optimization example using Gaussian-process Bayesian optimization with explicit safety and product-quality constraints.

The process is a stylized continuous stirred-tank reactor (CSTR) with two parallel first-order reactions:

```text
A -> B   desired product
A -> C   undesired by-product
```

Reaction rates follow Arrhenius kinetics. The model is intentionally compact enough for reproducible optimization experiments; it is not a calibrated chemical-plant digital twin.

## Decision variables

The optimizer selects:

| Variable | Range |
|---|---:|
| Reactor temperature | 360–440 K |
| Residence time | 0.2–2.0 h |
| Feed concentration | 0.8–2.0 kmol/m³ |
| Coolant intensity | 0.2–1.0 |

## Objective

Maximize hourly operating profit:

```text
product revenue
- feed cost
- by-product disposal cost
- sensible-heating energy cost
- coolant operating cost
```

## Constraints

The process must satisfy:

1. reaction heat generation ≤ heat-removal capacity;
2. by-product fraction among reacted material ≤ 0.22.

Both constraints are modeled as `g(x) <= 0`.

## Noisy experiments

The Bayesian optimizer does not observe the deterministic process model directly. Each experiment contains configurable noise in:

- profit;
- heat-balance measurement;
- by-product fraction measurement.

This separates the hidden benchmark process from the data available to the optimizer.

## Bayesian optimization

The implementation uses one Gaussian process for the objective and one for each constraint.

The acquisition function is constrained Expected Improvement:

```text
cEI(x) = EI(x) * P(g1(x) <= 0) * P(g2(x) <= 0)
```

Candidate selection uses Latin-hypercube exploration of the bounded operating region.

The final recommendation is deliberately more conservative than simply taking the largest noisy measured profit. It requires a configurable posterior probability of feasibility (95% by default in the benchmark reporting).

## Independent validation

The repository contains three reference mechanisms:

- dense Sobol quasi-random search;
- constrained Differential Evolution;
- a structural reference candidate exploiting the analytical selectivity-temperature relationship.

For the fixed model constants, the by-product specification implies a temperature boundary near:

```text
392.395906 K
```

The structural reference and constrained global search agree near:

```text
temperature        392.395906 K
residence time       0.743766 h
feed concentration   2.000000 kmol/m³
coolant intensity    0.340694
profit             ~1455.989 $/h
```

This agreement is strong validation of the implementation, but it is not presented as a mathematical proof of global optimality.

## Reference benchmark

With seed 42, 12 initial experiments and 24 BO iterations (36 expensive evaluations total), the validated run produced approximately:

```text
Conservative BO recommendation
posterior P(feasible)        0.987
true benchmark feasible      True
true benchmark profit        1364.74 $/h

Best random-search point
true benchmark profit        1054.99 $/h

Independent reference
profit                        1455.99 $/h
```

These numbers are reproducible for the pinned model and seed. They are an educational benchmark, not evidence that Bayesian optimization universally dominates random search.

A five-seed development check with a 30-evaluation budget had BO outperform the matched-budget random-search run in all five sampled seeds. That result is sample evidence only and is not a general performance guarantee.

## Run

```bash
python constrained_bayesian_optimization_cstr.py
```

Fast internal checks:

```bash
python constrained_bayesian_optimization_cstr.py --self-test
python -m unittest discover -s tests -v
```

## Scope and limitations

The reactor model uses ideal steady-state CSTR balances and simple parallel first-order kinetics. It does not include rigorous thermodynamics, transport limitations, controller dynamics, catalyst deactivation, pressure effects, multi-phase behavior, or plant-calibrated economics.

The Gaussian-process constraint models are treated as conditionally independent when calculating joint probability of feasibility.

The global reference is approximate. No global optimality certificate is claimed.
