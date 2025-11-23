#!/usr/bin/env python3
from vireon_sat import CNF, VireonTRPSolver
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dimacs", type=str)
    ap.add_argument("--max_steps", type=int, default=500000)
    args = ap.parse_args()

    cnf = CNF.from_dimacs(args.dimacs)
    solver = VireonTRPSolver(max_steps=args.max_steps)
    res = solver.solve(cnf)
    print(res["status"], "steps=", res["steps"], "best_unsat=", res["best_unsat"])
