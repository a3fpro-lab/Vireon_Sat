# Vireon_Sat
VIREON-SAT is Vireon’s first formal P vs NP probe: a structure-collapse 3-SAT solver with falsifiable matched-budget demos (no proof claims, just measurable wins or deaths)
# VIREON-SAT (v0)

VIREON-SAT is an experimental TRP/entropy-collapse solver for **3-SAT** (NP-complete).
It does **not** claim a P vs NP proof. The goal is simple: build a Vireon-native
structure-extraction solver and test it under pre-registered, matched-budget gates.

## Core idea
We define a Vireon potential on assignments \(x \in \{\pm1\}^n\):

\[
E(x)=\#\text{unsatisfied clauses} + \lambda \sum_i H(p_i),
\quad
p_i=\frac{d_i}{d_i+s_i+\epsilon}
\]

- \(d_i\): # unsatisfied clauses containing variable \(i\)  
- \(s_i\): # satisfied clauses containing \(i\)  
- \(H(p)\): binary entropy  
- \(\lambda\): fixed global drag weight

TRP collapse step chooses flips minimizing:

\[
\Delta_i/(w_i+\delta),
\quad
w_i=p_i(1-p_i)
\]

with a controlled micro-jitter when no downhill move exists.

## What’s in this repo
- `VireonTRPSolver`: main solver
- Baselines: `GreedyGSAT`, `WalkSAT`
- DIMACS 3-CNF parser
- Random phase-transition 3-SAT generator
- Demo harness + CI tests
- Pre-registered demo protocol template

## Install
```bash
pip install -e .
