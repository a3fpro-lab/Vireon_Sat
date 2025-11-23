from vireon_sat import CNF, VireonTRPSolver

def test_solver_finds_solution_small():
    cnf = CNF(3, [(1, 2, 3), (-1, 2, 3)])
    sol = VireonTRPSolver(max_steps=5000, seed=1)
    res = sol.solve(cnf)
    assert res["best_unsat"] == 0
