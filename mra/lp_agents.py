import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

from mra.utils import *
from mra.config import *


"""
Methods for RA problem
    minimize      f_1(x)+...+f_N(x) = \sum_i (c^T)_i x_i
    subject to    A x \leq b
                
using price interface
    minimize      \sum_{i=1}^N (c^T)_i x_i + I(-R \leq x_i \leq R)
    subject to    Ax \leq b
                  

"""


def lp_query_multiple_actions_noisy_prices(lamb, Ai, ci, R, num_points=1, percent=1e-2, eps=0):
    # introduce noise in prices proportional to each price
    # return K noisy actions (f_i(x_i) + (lambda - delta)^T x_i)
    n = ci.size
    yi = Ai.T @ lamb[:, np.newaxis]
    xi = cp.Variable((n, 1)) 
    f = cp.sum(ci.T @ xi) 
    if eps > 0:
        f += eps * cp.sum_squares(xi)
    new_yi = cp.Parameter((n, 1))
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ new_yi))), 
                      [cp.abs(xi) <= R])
    new_yi.value = yi
    prob.solve(solver=global_solver)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        l = -percent * np.abs(yi)
        u = yi - l  # yi + eps * |yi|
        l += yi     # yi - eps * |yi|
        assert (l <= yi).all() and (yi<= u).all(), \
            print(np.concatenate([l, yi, u], axis=1))
        noisy_yi = l + np.multiply(np.random.rand(yi.shape[0], num_points), u - l )

        for t in range(num_points-1):
            new_yi.value = noisy_yi[:, t:t+1]
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


def lp_query_multiple_actions(lamb, Ai, ci, R, num_points=1, eps_sublevel=1e-2, eps=0):
    # return x s.t. -\lambda \in \partial f_i(x_i)) +  N(-R <= x_i <= R)
    # return K-1 noisy actions s.t. x is on the boundary of 
    # eps-sublevel set of (f_i(x_i) + lambda^T x_i)
    n = ci.size
    yi = Ai.T @ lamb
    xi = cp.Variable((n, 1)) 
    f = cp.sum(ci.T @ xi) 
    if eps > 0:
        f += eps * cp.sum_squares(xi)
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi))), 
                      [cp.abs(xi) <= R])
    prob.solve(solver=global_solver)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        p_star = prob.value
        ni = cp.Parameter((n, 1))
        f = cp.sum(ci.T @ xi)
        if eps > 0:
            f += eps * cp.sum_squares(xi)
        g = cp.sum(f + cp.sum(xi.T @ yi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)), 
                      [cp.abs(xi) <= R, 
                       g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8])
        ni_val = np.random.randn(n, num_points-1)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
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


def centralized_solution_lps(A, b, c, R=None, eps=0):
    """
    Solve
        minimize      \sum_i ci^T xi
        subject to    Ax \leq b
    """
    m, n = A.shape
    x = cp.Variable((n, 1))
    constraints = [A @ x <= b]
    if not ( R is None or R == np.inf):
        constraints += [cp.abs(x) <= R]
    f = cp.sum(c.T @ x)
    if eps > 0:
        f += eps * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(f), constraints)
    prob.solve(solver=global_solver)
    true_lamb = constraints[0].dual_value
    true_f = f.value
    true_x = x.value
    return true_x, true_f, true_lamb


def lp_obj_value_pr(x_k, c, R, eps=0):
    f = 0 
    if type(x_k) is list:
        count  = 0
        for i in range(len(x_k)):
            if (np.abs(x_k[i]) > R + 1e-6).any():
                return np.inf
            ci = c[count:count+x_k[i].size]
            f += ci.T @ x_k[i] 
            count += x_k[i].size
            if eps > 0:
                f += eps * np.linalg.norm(x_k[i])**2
    else:
        if (np.abs(x_k) > R + 1e-6).any():
            return np.inf
        f = c.T @ x_k
        if eps > 0:
            f += eps * np.linalg.norm(x_k)**2
    return np.sum(f)


def lps_data(m, n, ni, num_agents):
    s0 = np.random.randn(m)
    lamb0 = np.maximum(-s0, 0) + np.random.rand(m)
    s0 = np.maximum(s0, 0)
    x0 = np.random.randn(n)
    A = np.random.randn(m, n)
    b = (A @ x0 + s0).reshape(-1, 1)
    c = np.random.randn(n) / 5 + -A.T @ lamb0 / 10

    R = 2 * np.linalg.norm(x0)

    c_all = []
    A_all = []
    for i in  range(num_agents):
        c_all += [c[ni*i : ni*(i+1),]]
        A_all += [A[:, ni*i : ni*(i+1),]]

    assert np.allclose(c, np.concatenate(c_all, axis=0)) and np.allclose(A, np.concatenate(A_all, axis=1))
    return c_all, A_all, A, b, c, R