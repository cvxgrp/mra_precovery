import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

from mra.utils import *
from mra.config import *


"""
Methods for RA problem
    minimize      f_1(x)+...+f_N(x) = \sum_i ci^T x
    subject to    A_i x \leq b_i
                
using consenus ADMM
    minimize      \sum_{i=1}^N c_i^T x_i + I(A_i x_i \leq b_i)
    subject to    x_i = z,  for all i=1, ..., N
                  

"""

def prox_lp_query_noisy_price_percent(yi, zi, Ai, bi, ci, rho, num_points=1, percent=None):
    # return x_i = \argmin(f_i(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    # introduce noise in prices proportional to each price
    m, n = Ai.shape
    xi = cp.Variable((n, 1))
    f = cp.sum(ci.T @ xi) 
    
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [Ai @ xi <= bi])
    prob.solve(solver=global_solver)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        # noise proportional to the price: [yi +/- eps * |yi|]
        l = -percent * np.abs(yi)
        u = yi - l 
        l += yi
        deltas = l + np.multiply(np.random.rand(n, num_points), u - l )
        new_yi = cp.Parameter((yi.shape[0], 1))
        prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ new_yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [Ai @ xi <= bi])
        for t in range(num_points-1):
            new_yi.value = deltas[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_lp_query(yi, zi, Ai, bi, ci, rho, num_points=1, deltas=None):
    # return x_i = \argmin(f_i(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    m, n = Ai.shape
    xi = cp.Variable((n, 1))
    f = cp.sum(ci.T @ xi) 
    
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [Ai @ xi <= bi])
    prob.solve(solver=global_solver)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        new_yi = cp.Parameter((yi.shape[0], 1))
        prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ new_yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [Ai @ xi <= bi])
        for t in range(num_points-1):
            new_yi.value = yi + deltas[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_lp_query_eps_o(yi, zi, Ai, bi, ci, rho, num_points=1, eps_sublevel=1e-2):
    # return x_i in the eps-sublevel set of (f_i(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    m, n = Ai.shape
    xi = cp.Variable((n, 1))
    f = cp.sum(ci.T @ xi) 
    
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [Ai @ xi <= bi])
    prob.solve(solver=global_solver)

    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        p_star = prob.value
        ni = cp.Parameter((n, 1))
        f = cp.sum(ci.T @ xi)
        g = cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)), 
                      [Ai @ xi <= bi, 
                       g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8])
        ni_val = np.random.randn(n, num_points-1)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
        assert np.allclose(np.linalg.norm(ni_val, axis=0), np.ones(num_points-1))
        for t in range(num_points-1):
            ni.value = ni_val[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [xi.value]
            assert prob.status == "optimal", print(prob.status)
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_lp_query_eps_conj_subgrad(yi, zi, Ai, bi, ci, R, rho, num_points=1, eps_sublevel=1e-2):
    # return multiple points wrt price interface
    # return x_i in the eps-sublevel set of (f_i(x_i)) + y_i^Tx_i)
    m, n = Ai.shape
    xi = cp.Variable((n, 1))
    f = cp.sum(ci.T @ xi) 
    
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [Ai @ xi <= bi, cp.abs(xi) <= R])
    prob.solve(solver=global_solver)

    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi))), 
                      [Ai @ xi <= bi, cp.abs(xi) <= R])
        prob.solve(solver=global_solver)
        p_star = prob.value 
        ni = cp.Parameter((n, 1))
        g = cp.sum(f + cp.sum(xi.T @ yi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)), 
                      [Ai @ xi <= bi, 
                       g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8,
                       cp.abs(xi) <= R])
        ni_val = np.random.randn(n, num_points-1)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
        assert np.allclose(np.linalg.norm(ni_val, axis=0), np.ones(num_points-1))
        for t in range(num_points-1):
            ni.value = ni_val[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [xi.value]
            assert prob.status == "optimal", print(prob.status)
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_lp_query_eps_o_price(yi, zi, Ai, bi, ci, rho, num_points=1, eps_sublevel=1e-2):
    # return multiple points wrt price interface
    # return x_i in the eps-sublevel set of (f_i(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    m, n = Ai.shape
    xi = cp.Variable((n, 1))
    f = cp.sum(ci.T @ xi) 
    
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [Ai @ xi <= bi])
    prob.solve(solver=global_solver)

    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        r = np.sqrt(cp.sum_squares(xi - zi).value)
        p_star = prob.value - (rho/2) * r**2
        ni = cp.Parameter((n, 1))
        g = cp.sum(f + cp.sum(xi.T @ yi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)), 
                      [Ai @ xi <= bi, 
                       g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8,
                       cp.sum_squares(xi - zi) <= 100*r**2])
        ni_val = np.random.randn(n, num_points-1)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
        assert np.allclose(np.linalg.norm(ni_val, axis=0), np.ones(num_points-1))
        for t in range(num_points-1):
            ni.value = ni_val[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [xi.value]
            assert prob.status == "optimal", print(prob.status)
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def centralized_solution_lp_admm(A_all, b_all, c_all, E):
    """
    Solve
        minimize      \sum_i ci^T xi 
        subject to    x = E z
                      Ai xi - bi \leq 0
    """
    m, n = A_all[0].shape
    num_agents = len(A_all)
    x = cp.Variable((E.shape[0], 1))
    z = cp.Variable((E.shape[1], 1))
    f = 0 
    constraints = [ x == E @ z, 
                    ]
    for i in range(num_agents):
        xi = x[i * n : (i+1) * n]
        f += cp.sum(c_all[i].T @ xi)
        constraints += [A_all[i] @ xi <= b_all[i]]
    prob = cp.Problem(cp.Minimize(f), constraints)
    prob.solve(solver=global_solver)
    true_lamb = constraints[0].dual_value
    true_f = f.value
    true_x = x.value
    return true_x, true_f, true_lamb, z.value


def lp_obj_value(x_k, A_all, b_all, c_all):
    f = 0 
    if type(x_k) is list:
        for i in range(len(A_all)):
            f += c_all[i].T @ x_k[i] 
    else:
        count  = 0
        for i in range(len(A_all)):
            xi = x_k[count:count+A_all[i].shape[1]]
            f += c_all[i].T @ xi 
            count += A_all[i].shape[1]
        assert count == x_k.shape[0]
    return np.sum(f)
