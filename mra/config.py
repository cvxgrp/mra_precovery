import cvxpy as cp

global_solver = cp.MOSEK

global_solver_options = {
                    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-4,    # Primal feasibility tolerance
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-4,    # Dual feasibility tolerance
                    'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-5,   # Reduction in the complementarity
                    'MSK_IPAR_INTPNT_MAX_ITERATIONS': 5000    # Increase max iterations
                }