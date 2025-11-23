#!/usr/bin/env python3
from vireon_sat import CNF, VireonTRPSolver, GreedyGSAT, WalkSAT
import numpy as np
import argparse

def demo(n_vars=200, ratio=4.2, trials=10, max_steps=200000, seed=1234):
    rng = np.random.default_rng(seed)

    vireon = VireonTRPSolver(max_steps=max_steps, seed=int(rng.integers(0, 2**32-1)))
    gsat   = GreedyGSAT(max_steps=max_steps, seed=int(rng.integers(0, 2**32-1)))
    walks  = WalkSAT(max_steps=max_steps, seed=int(rng.integers(0, 2**32-1)))

    results = {"vireon": [], "gsat": [], "walksat": []}

    for k in range(trials):
        cnf = CNF.random_3sat(n_vars, int(ratio*n_vars), seed=int(rng.integers(0, 2**32-1)))
        r_v = vireon.solve(cnf)
        r_g = gsat.solve(cnf)
        r_w = walks.solve(cnf)

        results["vireon"].append(r_v)
        results["gsat"].append(r_g)
        results["walksat"].append(r_w)

        print(f"[trial {k+1}/{trials}] "
              f"V:{r_v['status']}@{r_v['steps']} "
              f"G:{r_g['status']}@{r_g['steps']} "
              f"W:{r_w['status']}@{r_w['steps']}")

    def summarize(key):
        steps = np.array([r["steps"] for r in results[key]])
        sat = np.array([r["status"] == "SAT" for r in results[key]])
        return int(np.median(steps)), float(sat.mean())

    mv, sv = summarize("vireon")
    mg, sg = summarize("gsat")
    mw, sw = summarize("walksat")

    print("\n--- matched-budget summary ---")
    print(f"VIREON  median_steps={mv:7d}, solve_rate={sv:.2f}")
    print(f"GSAT    median_steps={mg:7d}, solve_rate={sg:.2f}")
    print(f"WalkSAT median_steps={mw:7d}, solve_rate={sw:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ratio", type=float, default=4.2)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    demo(args.n, args.ratio, args.trials, args.max_steps, args.seed)
