import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

from mra.utils import *
from mra.config import *




def lp_single_query_multiple_actions(lamb, A, c, b, num_points=1, eps_sublevel=1e-2, eps=0):
    # return x s.t. -\lambda \in \partial f_i(x_i))
    # return K-1 noisy actions s.t. x is on the boundary of 
    # eps-sublevel set of (f_i(x_i) + lambda^T x_i)
    n = c.size
    yi = A.T @ lamb
    xi = cp.Variable((n, 1)) 
    f = cp.sum(c.T @ xi) 
    if eps > 0:
        f += eps * cp.sum_squares(xi)
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi))), [A @ xi <= b])
    prob.solve(solver=global_solver)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        p_star = prob.value
        ni = cp.Parameter((n, 1))
        if eps > 0:
            f += eps * cp.sum_squares(xi)
        g = cp.sum(f + cp.sum(xi.T @ yi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)), 
                      [A @ xi <= b, 
                       g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8])
        ni_val = np.random.randn(n, num_points-1)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
        assert np.allclose(np.linalg.norm(ni_val, axis=0), np.ones(num_points-1))
        for t in range(num_points-1):
            ni.value = ni_val[:, t:t+1]
            try: prob.solve(solver=global_solver)
            except:
                try: prob.solve(solver="CLARABEL", tol_gap_rel=1e-3, tol_feas=1e-3, max_iter=100)
                except: prob.solve(solver="CLARABEL", tol_gap_rel=1e-1, tol_feas=1e-1, max_iter=100)
            xs += [xi.value]
            assert prob.status == "optimal", print(prob.status)
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def lpR_subgrad(x_k, c, R):
    # subgradient of (c^Tx + I(\|x\|_\infty \leq R))
    max_idx = np.argmax(np.abs(x_k))
    assert np.allclose(np.linalg.norm(x_k.flatten(), ord=np.inf), np.abs(x_k[max_idx])), print(np.linalg.norm(x_k.flatten(), ord=np.inf), np.abs(x_k[max_idx]))
    if np.abs(x_k[max_idx]) < R:
        return c 
    else:
        g = np.zeros(x_k.shape)
        g[max_idx] = np.sign(x_k[max_idx])
        return c + g
    
