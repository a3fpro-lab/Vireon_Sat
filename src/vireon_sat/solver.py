# -*- coding: utf-8 -*-
"""
VIREON-SAT v0
-------------
Deterministic TRP-style collapse solver for 3-SAT with a Vireon potential:
    E(x) = (# unsatisfied clauses) + λ * Σ_i H(p_i)
where p_i = d_i / (d_i + s_i + eps)

Library-only module for import.
Dependencies: numpy only.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


# -----------------------------
# CNF representation utilities
# -----------------------------

@dataclass
class CNF:
    n_vars: int
    clauses: List[Tuple[int, int, int]]  # 3-literal clauses (ints in ±[1..n])

    @staticmethod
    def from_dimacs(path: str) -> "CNF":
        clauses = []
        n_vars = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("c"):
                    continue
                if line.startswith("p"):
                    parts = line.split()
                    # p cnf n_vars n_clauses
                    n_vars = int(parts[2])
                    continue
                lits = [int(x) for x in line.split()]
                if not lits:
                    continue
                cur = []
                for lit in lits:
                    if lit == 0:
                        if len(cur) != 3:
                            raise ValueError(f"Need 3-SAT clauses, got {cur}")
                        clauses.append(tuple(cur))
                        cur = []
                    else:
                        cur.append(lit)
        if n_vars is None:
            raise ValueError("DIMACS header not found.")
        return CNF(n_vars=n_vars, clauses=clauses)

    @staticmethod
    def random_3sat(n_vars: int, m_clauses: int, seed: Optional[int] = None) -> "CNF":
        rng = np.random.default_rng(seed)
        clauses = []
        for _ in range(m_clauses):
            vs = rng.choice(np.arange(1, n_vars + 1), size=3, replace=False)
            signs = rng.choice([-1, 1], size=3, replace=True)
            clause = tuple(int(s * v) for s, v in zip(signs, vs))
            clauses.append(clause)
        return CNF(n_vars=n_vars, clauses=clauses)


# -----------------------------
# Core SAT state + bookkeeping
# -----------------------------

class SATState:
    """
    Incremental SAT bookkeeping for fast flip evaluation.

    Assignment x in {+1, -1}^n
    Clauses satisfied if any literal true.
    """

    def __init__(self, cnf: CNF, x0: Optional[np.ndarray] = None, seed: Optional[int] = None):
        self.cnf = cnf
        self.n = cnf.n_vars
        self.m = len(cnf.clauses)
        self.rng = np.random.default_rng(seed)

        # assignment in ±1
        if x0 is None:
            self.x = self.rng.choice([-1, 1], size=self.n).astype(np.int8)
        else:
            self.x = x0.astype(np.int8).copy()

        # Precompute clause var indices and signs
        self.c_vars = np.zeros((self.m, 3), dtype=np.int32)   # 0-based var indices
        self.c_sign = np.zeros((self.m, 3), dtype=np.int8)   # ±1
        for j, (a, b, c) in enumerate(cnf.clauses):
            lits = (a, b, c)
            for k, lit in enumerate(lits):
                v = abs(lit) - 1
                s = 1 if lit > 0 else -1
                self.c_vars[j, k] = v
                self.c_sign[j, k] = s

        # Adjacency: for each var, list of clauses containing it
        self.var_to_clauses: List[List[int]] = [[] for _ in range(self.n)]
        for j in range(self.m):
            for k in range(3):
                self.var_to_clauses[self.c_vars[j, k]].append(j)

        # Clause sat status + #true literals per clause
        self.true_count = np.zeros(self.m, dtype=np.int8)
        self.satisfied = np.zeros(self.m, dtype=bool)

        # For each var, d_i (#unsat clauses containing var) and s_i (#sat clauses containing var)
        self.d = np.zeros(self.n, dtype=np.int32)
        self.s = np.zeros(self.n, dtype=np.int32)

        self._initialize_counts()

    def _initialize_counts(self):
        # Compute satisfaction per clause and fill d/s
        self.true_count[:] = 0
        self.satisfied[:] = False
        for j in range(self.m):
            t = 0
            for k in range(3):
                v = self.c_vars[j, k]
                sgn = self.c_sign[j, k]
                # literal true iff sgn * x_v == +1
                if sgn * self.x[v] == 1:
                    t += 1
            self.true_count[j] = t
            self.satisfied[j] = (t > 0)

        self.d[:] = 0
        self.s[:] = 0
        for v in range(self.n):
            for j in self.var_to_clauses[v]:
                if self.satisfied[j]:
                    self.s[v] += 1
                else:
                    self.d[v] += 1

    def unsat_clauses(self) -> np.ndarray:
        return np.where(~self.satisfied)[0]

    def num_unsat(self) -> int:
        return int((~self.satisfied).sum())

    def flip(self, v: int):
        """
        Flip variable v (0-based), update all affected bookkeeping.
        """
        old_val = self.x[v]
        new_val = -old_val
        self.x[v] = new_val

        # Update clauses containing v
        for j in self.var_to_clauses[v]:
            old_sat = self.satisfied[j]
            old_true = self.true_count[j]

            # recompute literal truth for v inside clause j only
            delta_true = 0
            for k in range(3):
                if self.c_vars[j, k] == v:
                    sgn = self.c_sign[j, k]
                    old_lit_true = (sgn * old_val == 1)
                    new_lit_true = (sgn * new_val == 1)
                    delta_true = int(new_lit_true) - int(old_lit_true)
                    break

            new_true = old_true + delta_true
            new_sat = (new_true > 0)

            if old_sat == new_sat:
                self.true_count[j] = new_true
                continue

            # Satisfaction toggled: update variable d/s for all vars in clause
            self.true_count[j] = new_true
            self.satisfied[j] = new_sat

            vars_in_clause = self.c_vars[j]
            if new_sat and (not old_sat):
                # clause moved unsat -> sat
                for u in vars_in_clause:
                    self.d[u] -= 1
                    self.s[u] += 1
            elif (not new_sat) and old_sat:
                # clause moved sat -> unsat
                for u in vars_in_clause:
                    self.s[u] -= 1
                    self.d[u] += 1

    def delta_unsat_if_flip(self, v: int) -> int:
        """
        Return change in number of unsatisfied clauses if v were flipped.
        Only scans clauses containing v.
        """
        old_val = self.x[v]
        new_val = -old_val
        delta_unsat = 0

        for j in self.var_to_clauses[v]:
            old_true = self.true_count[j]
            old_sat = (old_true > 0)

            delta_true = 0
            for k in range(3):
                if self.c_vars[j, k] == v:
                    sgn = self.c_sign[j, k]
                    old_lit_true = (sgn * old_val == 1)
                    new_lit_true = (sgn * new_val == 1)
                    delta_true = int(new_lit_true) - int(old_lit_true)
                    break

            new_true = old_true + delta_true
            new_sat = (new_true > 0)

            if old_sat and not new_sat:
                delta_unsat += 1
            elif (not old_sat) and new_sat:
                delta_unsat -= 1

        return delta_unsat


# -----------------------------
# Vireon potential + TRP solver
# -----------------------------

class VireonTRPSolver:
    """
    Vireon TRP collapse solver for 3-SAT.

    Step rule:
      p_i = d_i/(d_i+s_i+eps)
      w_i = p_i*(1-p_i)

      Δ_i = E(flip i) - E
      choose i* = argmin Δ_i/(w_i+δ)
      if Δ_i* < 0: flip i*
      else TRP micro-jitter: sample among top-K ratios via softmax(T_trp)
    """

    def __init__(
        self,
        lam: float = 0.05,
        eps: float = 1e-9,
        delta: float = 1e-6,
        top_k: int = 8,
        T_trp: float = 0.25,
        max_steps: int = 200_000,
        seed: Optional[int] = None,
    ):
        self.lam = float(lam)
        self.eps = float(eps)
        self.delta = float(delta)
        self.top_k = int(top_k)
        self.T_trp = float(T_trp)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)

    def _entropy_binary(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, 1.0 - self.eps)
        return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)

    def energy(self, st: SATState) -> float:
        unsat = st.num_unsat()
        denom = st.d + st.s + self.eps
        p = st.d / denom
        drag = self._entropy_binary(p).sum()
        return float(unsat + self.lam * drag)

    def solve(self, cnf: CNF, x0: Optional[np.ndarray] = None) -> Dict:
        st = SATState(cnf, x0=x0, seed=int(self.rng.integers(0, 2**32 - 1)))

        best_unsat = st.num_unsat()
        best_x = st.x.copy()
        trace = []

        # precompute for speed
        H_cur_cache = None
        p_cache = None

        for t in range(self.max_steps):
            unsat_idx = st.unsat_clauses()
            nu = len(unsat_idx)
            if nu == 0:
                return {
                    "status": "SAT",
                    "steps": t,
                    "assignment": st.x.copy(),
                    "best_unsat": 0,
                    "trace": trace,
                }

            denom = st.d + st.s + self.eps
            p = st.d / denom
            w = p * (1.0 - p)

            # cache entropy for this step
            H_cur = self._entropy_binary(p)
            H_cur_cache = H_cur
            p_cache = p

            # candidate vars: those touching unsatisfied clauses
            cand = set()
            for j in unsat_idx:
                cand.update(st.c_vars[j])
            cand = np.fromiter(cand, dtype=np.int32)
            if cand.size == 0:
                cand = np.arange(st.n, dtype=np.int32)

            ratios = np.empty(cand.size, dtype=np.float64)
            deltas = np.empty(cand.size, dtype=np.float64)

            cur_unsat = nu

            for idx, v in enumerate(cand):
                v = int(v)
                delta_unsat = st.delta_unsat_if_flip(v)
                new_unsat = cur_unsat + delta_unsat

                # find toggled clauses under hypothetical flip of v
                affected_clauses = st.var_to_clauses[v]
                toggled = []
                old_val = st.x[v]
                new_val = -old_val

                for j in affected_clauses:
                    old_true = st.true_count[j]
                    old_sat = old_true > 0

                    delta_true = 0
                    for k in range(3):
                        if st.c_vars[j, k] == v:
                            sgn = st.c_sign[j, k]
                            old_lit_true = (sgn * old_val == 1)
                            new_lit_true = (sgn * new_val == 1)
                            delta_true = int(new_lit_true) - int(old_lit_true)
                            break
                    new_true = old_true + delta_true
                    new_sat = new_true > 0
                    if old_sat != new_sat:
                        toggled.append(j)

                # local update for drag term
                d_new = st.d.copy()
                s_new = st.s.copy()
                for j in toggled:
                    vars_in_clause = st.c_vars[j]
                    if st.satisfied[j] is False:
                        # would become sat
                        for u in vars_in_clause:
                            d_new[u] -= 1
                            s_new[u] += 1
                    else:
                        # would become unsat
                        for u in vars_in_clause:
                            s_new[u] -= 1
                            d_new[u] += 1

                denom_new = d_new + s_new + self.eps
                p_new = d_new / denom_new
                H_new = self._entropy_binary(p_new)

                delta_drag = (H_new - H_cur_cache).sum()
                delta_E = (new_unsat - cur_unsat) + self.lam * delta_drag

                deltas[idx] = delta_E
                ratios[idx] = delta_E / (w[v] + self.delta)

            best_pos = int(np.argmin(ratios))
            best_i = int(cand[best_pos])
            best_delta = float(deltas[best_pos])

            if best_delta < 0:
                st.flip(best_i)
            else:
                k = min(self.top_k, cand.size)
                top_idx = np.argpartition(ratios, k - 1)[:k]
                top_vars = cand[top_idx]
                top_ratios = ratios[top_idx]
                logits = -top_ratios / max(self.T_trp, 1e-9)
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                v_choice = int(self.rng.choice(top_vars, p=probs))
                st.flip(v_choice)

            nu2 = st.num_unsat()
            if nu2 < best_unsat:
                best_unsat = nu2
                best_x = st.x.copy()

            if (t % 1000) == 0:
                trace.append((t, nu2, best_unsat))

        return {
            "status": "UNKNOWN",
            "steps": self.max_steps,
            "assignment": best_x,
            "best_unsat": best_unsat,
            "trace": trace,
        }


# -----------------------------
# Baselines for matched budgets
# -----------------------------

class GreedyGSAT:
    """Flip var giving best unsat decrease; random tie-break."""
    def __init__(self, max_steps=200_000, seed=None):
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)

    def solve(self, cnf: CNF) -> Dict:
        st = SATState(cnf, seed=int(self.rng.integers(0, 2**32 - 1)))
        best_unsat = st.num_unsat()
        best_x = st.x.copy()

        for t in range(self.max_steps):
            nu = st.num_unsat()
            if nu == 0:
                return {"status": "SAT", "steps": t, "best_unsat": 0, "assignment": st.x.copy()}

            unsat_idx = st.unsat_clauses()
            cand = set()
            for j in unsat_idx:
                cand.update(st.c_vars[j])
            cand = np.fromiter(cand, dtype=np.int32)
            if cand.size == 0:
                cand = np.arange(st.n, dtype=np.int32)

            best_delta = 10**9
            best_vars = []
            for v in cand:
                delta_unsat = st.delta_unsat_if_flip(int(v))
                if delta_unsat < best_delta:
                    best_delta = delta_unsat
                    best_vars = [int(v)]
                elif delta_unsat == best_delta:
                    best_vars.append(int(v))

            v_choice = int(self.rng.choice(best_vars))
            st.flip(v_choice)

            nu2 = st.num_unsat()
            if nu2 < best_unsat:
                best_unsat = nu2
                best_x = st.x.copy()

        return {"status": "UNKNOWN", "steps": self.max_steps, "best_unsat": best_unsat, "assignment": best_x}


class WalkSAT:
    """Standard WalkSAT-like heuristic."""
    def __init__(self, p_random=0.5, max_steps=200_000, seed=None):
        self.p_random = float(p_random)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)

    def solve(self, cnf: CNF) -> Dict:
        st = SATState(cnf, seed=int(self.rng.integers(0, 2**32 - 1)))
        best_unsat = st.num_unsat()
        best_x = st.x.copy()

        for t in range(self.max_steps):
            nu = st.num_unsat()
            if nu == 0:
                return {"status": "SAT", "steps": t, "best_unsat": 0, "assignment": st.x.copy()}

            unsat_idx = st.unsat_clauses()
            j = int(self.rng.choice(unsat_idx))
            vars_in_clause = st.c_vars[j]

            if self.rng.random() < self.p_random:
                v_choice = int(self.rng.choice(vars_in_clause))
            else:
                best_delta = 10**9
                best_vars = []
                for v in vars_in_clause:
                    delta_unsat = st.delta_unsat_if_flip(int(v))
                    if delta_unsat < best_delta:
                        best_delta = delta_unsat
                        best_vars = [int(v)]
                    elif delta_unsat == best_delta:
                        best_vars.append(int(v))
                v_choice = int(self.rng.choice(best_vars))

            st.flip(v_choice)

            nu2 = st.num_unsat()
            if nu2 < best_unsat:
                best_unsat = nu2
                best_x = st.x.copy()

        return {"status": "UNKNOWN", "steps": self.max_steps, "best_unsat": best_unsat, "assignment": best_x}
