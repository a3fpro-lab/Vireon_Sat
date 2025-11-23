---
name: Demo Protocol (Pre-Registered)
about: Lock a falsifiable demo gate before running it.
title: "[DEMO] V-SAT gate v0"
labels: demo, prereg
---

## Target
Canonical NP-complete problem: 3-SAT.

## Datasets
- Random 3-SAT near phase transition (m/n ≈ 4.2):
  - n ∈ {100, 200, 400, 800}
  - 20 instances per n
- Optional industrial SATLIB 3-CNF benchmarks (listed explicitly here before run).

## Solvers
- VIREON-SAT (this repo)
- GreedyGSAT baseline
- WalkSAT baseline
- CDCL baseline (external, named + version)

## Matched Budget
- Max flips/steps: __________
- Wall clock cap: __________
- Seeds: fixed list of ≥10 seeds (paste below)

## Metrics
Primary:
- median flips to SAT
- solve rate within budget

Secondary:
- mean flips, IQR
- best_unsat at timeout

## Win Condition
VIREON-SAT must show:
- ≥ 40% lower median flips than *best baseline*
- with ≥ baseline solve rate
under matched budgets.

## Falsifiers
- Speedup disappears on literal-shuffle controls.
- Requires per-instance tuning of λ or T_trp.
- Speedup vanishes when counting real ops instead of flips.
