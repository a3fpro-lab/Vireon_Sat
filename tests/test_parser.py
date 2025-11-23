from vireon_sat import CNF

def test_random_3sat_shapes():
    cnf = CNF.random_3sat(50, 200, seed=0)
    assert cnf.n_vars == 50
    assert len(cnf.clauses) == 200
    assert all(len(c) == 3 for c in cnf.clauses)
