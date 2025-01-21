import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

from mra.utils import *
from mra.config import *


"""
Methods for OT problem
    minimize      Tr( C^T X) 
    subject to    X^T 1 = b
                      X 1 = a
                      X >= 0
                
using price interface
    minimize      \sum_i ci^T xi + I(1 \geq x_i \geq 0) + I(x_i^T 1 = a_i)
    subject to    X^T 1 = b
                  X^Tv \leq c
                  
"""


def ot_optimal_action(ci, yi, ai, n):
    idx = np.argmin((ci + yi)[:,0])
    xi = np.zeros((n, 1))
    xi[idx, 0] = ai
    p_star = (xi.T @ (ci + yi)).sum()
    return xi, p_star


def ot_yi_lamb(lamb, ci, vol_i):
    if lamb.size == ci.size:
        yi = lamb[:, np.newaxis]
    else:
        yi =  vol_i * lamb[:ci.size, np.newaxis] + lamb[ci.size:, np.newaxis] 
    return yi


def ot_query_multiple_actions_noisy_prices(lamb, ci, ai, vol_i=None, num_points=1, percent=1e-2):
    # return x s.t. -\lambda \in \partial f_i(x_i)) 
    # introduce noise in prices proportional to each price
    # return K noisy actions (f_i(x_i) + (lambda - delta)^T x_i)
    n = ci.size
    
    yi = ot_yi_lamb(lamb, ci, vol_i)
    xi, p_star = ot_optimal_action(ci, yi, ai, n)

    if num_points == 1:
        return xi #, prob.value
    else:
        xs = [xi]
        l = -percent * np.abs(yi)
        u = yi - l  # yi + eps * |yi|
        l += yi     # yi - eps * |yi|
        assert (l <= yi).all() and (yi <= u).all()
        noisy_yi = l + np.multiply(np.random.rand(yi.size, num_points), u - l )
        for t in range(num_points-1):
            yi = ot_yi_lamb(noisy_yi[:, t], ci, vol_i)
            xi, p_star = ot_optimal_action(ci, yi, ai, n)
            xs += [xi]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def ot_query_multiple_actions(lamb, ci, ai, vol_i=None, num_points=1, eps_sublevel=1e-2):
    # return x s.t. -\lambda \in \partial f_i(x_i)) 
    # return K-1 noisy actions s.t. x is on the boundary of 
    # eps-sublevel set of (f_i(x_i) + lambda^T x_i)
    n = ci.size
    yi = ot_yi_lamb(lamb, ci, vol_i)
    xi, p_star = ot_optimal_action(ci, yi, ai, n)

    if num_points == 1:
        return xi #, prob.value
    else:
        xs = [xi]
        ni = cp.Parameter((n, 1))
        xi = cp.Variable((n, 1), nonneg=True)
        f = cp.sum(ci.T @ xi)
        g = cp.sum(f + cp.sum(xi.T @ yi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)), 
                      [xi <= 1, cp.sum(xi) == ai,
                       g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8])
        ni_val = np.random.randn(n, num_points-1)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
        # ni_val = orthog_nis(n, num_points-1)
        assert np.allclose(np.linalg.norm(ni_val, axis=0), np.ones(num_points-1))
        for t in range(num_points-1):
            ni.value = ni_val[:, t:t+1]
            try:
                prob.solve(solver=global_solver)
            except:
                try:
                    prob.solve(solver="CLARABEL", tol_gap_rel=1e-3, tol_feas=1e-3, max_iter=100)
                except:
                    prob.solve(solver="CLARABEL", tol_gap_rel=1e-1, tol_feas=1e-1, max_iter=100)
            xs += [xi.value]
            assert prob.status == "optimal", print(prob.status)
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def ot_obj_value(x, C, a, b):
    if type(x) is list:
        X = np.concatenate(x, axis=0)
    else:
        X = x.reshape(C.shape)
        assert np.allclose(X[0, :], x[:C.shape[1], 0])
    if not np.allclose(X.sum(axis=1), a[:,0]):
        return np.inf
    return np.sum(np.trace(C.T @ X))


def centralized_solution_ot(C, a, A_eq, b_eq, A_ineq=None, b_ineq=None):
    """
    Solve
        minimize      \sum_i ci^T xi + I(1 \geq x_i \geq 0) + I(x_i^T 1 = a_i)
        subject to    X^T 1 = b
    """
    m, n = C.shape
    x = cp.Variable((C.size, 1), nonneg=True)
    constraints = [x <= 1]
    f = 0
    for i in range(m):
        f += cp.sum(C[i:i+1] @ x[i*n : (i+1)*n])
        constraints += [cp.sum(x[i*n : (i+1)*n]) == a[i,0]]
    if A_ineq is not None:
        constraints += [A_ineq @ x <= b_ineq]
    constraints += [A_eq @ x == b_eq]
    prob = cp.Problem(cp.Minimize(f), constraints)
    prob.solve()#(solver=global_solver)
    if A_ineq is not None:
        true_lamb = np.concatenate([constraints[-2].dual_value,
                                    constraints[-1].dual_value], axis=0)
    else:
        true_lamb = constraints[-1].dual_value
    true_f = f.value
    true_x = x.value
    return true_x, true_f, true_lamb



def centralized_solution_ot_matrix(C, a, b, vol=None, cap=None):
    """
    Solve
        minimize      Tr(C^T X) 
        subject to    X 1 = a
                      X >= 0
                      X^T 1 = b
                      X^T v <= c
    """
    X = cp.Variable(C.shape, nonneg=True)
    constraints = [cp.sum(X, axis=0) == b[:,0], cp.sum(X, axis=1) == a[:,0], X <= 1]
    if vol is not None:
        constraints += [X.T @ vol <= cap]
    f = cp.sum(cp.trace(C.T @ X))
    prob = cp.Problem(cp.Minimize(f), constraints)
    prob.solve()#(solver=global_solver)
    true_lamb = constraints[0].dual_value
    true_f = f.value
    true_x = X.value
    return true_x, true_f, true_lamb


def ot_data(m, n):
    a = np.exp(np.random.randn(m, 1))
    a /= a.sum()

    b = np.exp(np.random.randn(n, 1))
    b /= b.sum()

    assert np.allclose(a.sum(), 1) and (a >= 0).all() and np.allclose(b.sum(), 1) and (b >= 0).all()

    sigma = 0.8
    volumes = np.exp(np.random.randn(m, 1)*sigma)
    mean = np.exp(sigma**2/2)
    A_ineq = np.concatenate([volumes[i, 0] * np.eye(n) for i in range(m)] , axis=1)
    b_ineq = mean * b

    A_eq = np.concatenate([np.eye(n)] * m, axis=1)
    b_eq = b

    dim = 10
    s = np.random.randn(m, dim)
    s /= np.linalg.norm(s, axis=1)[:, None]
    t = np.random.randn(n, dim)
    t /= np.linalg.norm(t, axis=1)[:, None]
    d_s = (s * s).sum(axis=1)[:, np.newaxis]
    d_t = (t * t).sum(axis=1)[:, np.newaxis]

    C = d_s @ np.ones((1, n)) - 2 * s @ t.T + np.ones((m ,1)) @ d_t.T
    assert np.allclose(C[2, 3], np.linalg.norm(s[2]-t[3])**2)
    assert np.allclose(np.linalg.norm(s, axis=1), 1) and np.allclose(np.linalg.norm(t, axis=1), 1)
    return C, a, A_eq, b_eq, A_ineq, b_ineq, volumes, mean


def ot_prime_initial_bound(true_lamb, A_eq, b_eq, A_ineq, b_ineq):
    price_max = true_lamb.max()
    price_min = true_lamb.min()

    price_max = price_max * 3 if price_max > 0 else price_max / 3
    price_min = price_min / 3 if price_min > 0 else price_min * 3

    dual_var_size = 0
    if A_eq is not None:
        dual_var_size += b_eq.size
    if A_ineq is not None:
        dual_var_size += b_ineq.size
    price_max = price_max * np.ones((dual_var_size, 1))
    price_min = price_min * np.ones((dual_var_size, 1))
    if A_ineq is not None:
        price_min[:b_ineq.size] = np.maximum(price_min[:b_ineq.size], 0)
    return price_min, price_max