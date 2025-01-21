import cvxpy as cp
import numpy as np





def cvxpy_analytic_center(A, b):
    m, n = A.shape
    x = cp.Variable((n, 1))
    f = - cp.sum(cp.log(b - A @ x + 1e-8))
    prob = cp.Problem(cp.Minimize(f))
    prob.solve()
    return x.value


def cvxpy_analytic_center_prox(A, b, p_k, alpha, solver=cp.MOSEK, dual_gap=1e-8, primal_gap=1e-8, verbose=False):
    # add prox term
    # p^{k+1} = \argmin_p \phi^k(p) + (alpha/2) \|p - p^k\|^2
    m, n = A.shape
    x = cp.Variable((n, 1))
    f = - cp.sum(cp.log(b - A @ x + 1e-8))
    if p_k is not None:
        f += (alpha / 2) * cp.sum_squares(x - p_k)
    prob = cp.Problem(cp.Minimize(f))
    solver_options = {'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': dual_gap,
                                           'MSK_DPAR_INTPNT_CO_TOL_PFEAS': primal_gap}}
    try:
        prob.solve(verbose=verbose, solver=solver, **solver_options)
    except:
        try:
            prob.solve(solver="CLARABEL", tol_gap_rel=1e-3, tol_feas=1e-3, max_iter=100)
        except:
            prob.solve(solver="CLARABEL", tol_gap_rel=0.5, tol_feas=0.5, max_iter=100, verbose=True)
    return x.value


def cvxpy_analytic_center_l2(A, b, alpha, solver=cp.MOSEK, dual_gap=1e-8, primal_gap=1e-8, verbose=False):
    # add l2 regularization term
    # p^{k+1} = \argmin_p \phi^k(p) + (alpha/2) \|p\|^2
    m, n = A.shape
    x = cp.Variable((n, 1))
    f = - cp.sum(cp.log(b - A @ x + 1e-8)) + (alpha / 2) * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(f))
    solver_options = {'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': dual_gap,
                                           'MSK_DPAR_INTPNT_CO_TOL_PFEAS': primal_gap}}
    try:
        prob.solve(verbose=verbose, solver=solver, **solver_options)
    except:
        try:
            prob.solve(solver="CLARABEL", tol_gap_rel=1e-3, tol_feas=1e-3, max_iter=100)
        except:
            prob.solve(solver="CLARABEL", tol_gap_rel=0.5, tol_feas=0.5, max_iter=100, verbose=True)
    return x.value


def cvxpy_analytic_center_l2_cone(A0, b0, alpha=1, start_idx=None, solver=cp.MOSEK, dual_gap=1e-8, primal_gap=1e-8, verbose=False):
    # add l2 regularization term
    # p^{k+1} = \argmin_p \phi^k(p) + (alpha/2) \|p\|^2
    m, n = A0.shape
    x = cp.Variable((n, 1))
    t = cp.Variable()
    if start_idx <= A0.shape[0]:
        norms = np.concatenate([np.ones((start_idx, 1)),
                               ((np.linalg.norm(A0[start_idx:], axis=1)[:, None])**2 + b0[start_idx:]**2)**0.5], axis=0)
        A = np.divide(A0, norms)
        b = np.divide(b0, norms)
    else:
        A, b = A0, b0
    f = - cp.sum(cp.log(b * t - A @ x + 1e-8)) - cp.log(t) + (alpha / 2) * cp.sum_squares(x) + (alpha / 2) * cp.power(t, 2)
    prob = cp.Problem(cp.Minimize(f))
    solver_options = {'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': dual_gap,
                                           'MSK_DPAR_INTPNT_CO_TOL_PFEAS': primal_gap}}
    try:
        prob.solve(verbose=verbose, solver=solver, **solver_options)
    except:
        try:
            prob.solve(solver="CLARABEL", tol_gap_rel=1e-3, tol_feas=1e-3, max_iter=100)
        except:
            prob.solve(solver="CLARABEL", tol_gap_rel=0.5, tol_feas=0.5, max_iter=100, verbose=True)
    return x.value / t.value


def mve_center(A, b, verbose=False, solver=cp.MOSEK, dual_gap=1e-8):
    m, n = A.shape
    B = cp.Variable((n, n))
    d = cp.Variable((n, 1))
    f = -cp.sum(cp.log(cp.diag(B)))
    constraints = [ cp.upper_tri(B) == np.zeros(((n-1) * n // 2, 1)) ]
    for i in range(m):
        constraints += [cp.sum(cp.norm( B @ A[i:i+1, :].T, 2) + A[i:i+1, :] @ d) <= b[i]]
    prob = cp.Problem(cp.Minimize(f), constraints=constraints)
    solver_options = {'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': dual_gap}}
    try:
        prob.solve(verbose=verbose, solver=solver, **solver_options)
    except:
        try:
            prob.solve(solver="CLARABEL", tol_gap_rel=1e-3, tol_feas=1e-3, max_iter=100)
        except:
            prob.solve(solver="CLARABEL", tol_gap_rel=1e-1, tol_feas=1e-1, max_iter=100)
    return d.value, prob.value


def mve_center_prox(A, b, p_k, alpha, verbose=False, solver=cp.MOSEK, dual_gap=1e-8, primal_gap=1e-8):
    # add prox term
    # p^{k+1} = \argmin_p \phi^k(p) + (alpha/2) \|p - p^k\|^2
    m, n = A.shape
    B = cp.Variable((n, n))
    d = cp.Variable((n, 1))
    f = -cp.sum(cp.log(cp.diag(B)))
    if p_k is not None and alpha > 0:
        f += (alpha / 2) * cp.sum_squares(d - p_k)
    constraints = [ cp.upper_tri(B) == np.zeros(((n-1) * n // 2, 1)) ]
    for i in range(m):
        constraints += [cp.sum(cp.norm( B @ A[i:i+1, :].T, 2) + A[i:i+1, :] @ d) <= b[i]]
    prob = cp.Problem(cp.Minimize(f), constraints=constraints)
    if solver == cp.MOSEK:
        solver_options = {'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': dual_gap,
                                           'MSK_DPAR_INTPNT_CO_TOL_PFEAS': primal_gap}}
    else:
        solver_options = {}
    try:
        prob.solve(verbose=verbose, solver=solver, **solver_options)
    except:
        prob.solve(verbose=verbose, solver=cp.CLARABEL)
    return d.value, prob.value


def mve_center_logdet(A, b, verbose=False, solver=cp.MOSEK, dual_gap=1e-8):
    m, n = A.shape
    B = cp.Variable((n, n), symmetric=True)
    d = cp.Variable((n, 1))
    f = -cp.log_det(B)
    constraints = [B >> 0]
    for i in range(m):
        constraints += [cp.sum(cp.norm( B @ A[i:i+1, :].T, 2) + A[i:i+1, :] @ d) <= b[i]]
    prob = cp.Problem(cp.Minimize(f), constraints=constraints)
    solver_options = {'mosek_params': {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': dual_gap}}
    prob.solve(verbose=verbose, solver=solver, **solver_options)
    return d.value, prob.value
