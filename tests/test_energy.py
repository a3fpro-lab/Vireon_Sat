import numpy as np
from vireon_sat import CNF, SATState, VireonTRPSolver

def test_energy_zero_for_satisfied():
    cnf = CNF(3, [(1, 2, 3)])
    st = SATState(cnf, x0=np.array([1, 1, 1]))
    sol = VireonTRPSolver(lam=0.0)
    assert sol.energy(st) == 0.0
