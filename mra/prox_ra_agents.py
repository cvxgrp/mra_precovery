import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

from mra.utils import *
from mra.config import *


"""
Methods for RA problem
    minimize      f_1(x_1)+...+f_N(x_N) = \sum_i -\geo_mean(A_i x_i - b_i)
    subject to    x_1+...+x_N \leq d
using consenus ADMM
    minimize      \sum_{i=1}^N -\geo_mean(A_i x_ii - b_i) + I(Cx_{N+1} \leq d)
    subject to    x_i,j = x_{N+1},j = z_j,  for all i,j=1, ..., N
                  

"""


def prox_ra_query_noisy_price_percent(yi, zi, Ai, bi, rho, num_points=1, eps=1e-4, percent=None):
    # return x_i = \argmin(f_i(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    # introduce noise in prices proportional to each price
    m, n = Ai.shape
    xi = cp.Variable((n, 1), nonneg=True)
    f = -cp.geo_mean(Ai @ xi + bi) 
    if eps > 0:
        f -= eps * cp.sum(cp.sqrt(xi))
    
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))))
    prob.solve(solver=global_solver)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        # noise proportional to the price: [yi +/- eps * |yi|] 
        l = -percent * np.abs(yi)
        u = yi - l # yi + eps * |yi|
        l += yi    # yi - eps * |yi|
        assert (l <= yi).all() and (yi <= u).all(), print(np.concatenate([l, yi, u], axis=1))
        deltas = l + np.multiply(np.random.rand(n, num_points), u - l )
        new_yi = cp.Parameter((yi.shape[0], 1))
        prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ new_yi) + (rho/2) * cp.sum_squares(xi - zi))))
        for t in range(num_points-1):
            new_yi.value = deltas[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_ra_g_query_noisy_price_percent(yi, zi, C, d, rho, num_points=1, percent=None):
    # g(x_i) = I(Cx \leq d)
    # return x_i = \argmin(g(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    # introduce noise in prices proportional to each price
    n = C.shape[1]
    xi = cp.Variable((n, 1), nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [C @ xi <= d])
    prob.solve(solver=global_solver)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        # noise proportional to the price: [yi +/- eps * |yi|]
        l = -percent * np.abs(yi)
        u = yi - l 
        l += yi
        deltas = l + np.multiply(np.random.rand(n, num_points), u -l )
        new_yi = cp.Parameter((yi.shape[0], 1))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.sum(xi.T @ new_yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [C @ xi <= d])
        for t in range(num_points-1):
            new_yi.value = deltas[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_ra_query(yi, zi, Ai, bi, rho, num_points=1, eps=1e-4, deltas=None):
    # return x_i = \argmin(f_i(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    m, n = Ai.shape
    xi = cp.Variable((n, 1), nonneg=True)
    f = -cp.geo_mean(Ai @ xi + bi) 
    if eps > 0:
        f -= eps * cp.sum(cp.sqrt(xi))
    
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))))
    prob.solve(solver=global_solver)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        new_yi = cp.Parameter((yi.shape[0], 1))
        prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ new_yi) + (rho/2) * cp.sum_squares(xi - zi))))
        for t in range(num_points-1):
            new_yi.value = yi + deltas[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_ra_g_query(yi, zi, C, d, rho, num_points=1, deltas=None):
    # g(x_i) = I(Cx \leq d)
    # return x_i = \argmin(g(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    n = C.shape[1]
    xi = cp.Variable((n, 1), nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [C @ xi <= d])
    prob.solve(solver=global_solver)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        new_yi = cp.Parameter((yi.shape[0], 1))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.sum(xi.T @ new_yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [C @ xi <= d])
        for t in range(num_points-1):
            new_yi.value = yi + deltas[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def centralized_solution_ra_admm(A_all, b_all, A_G, C, d, eps=1e-4):
    """
    Solve
        minimize      \sum_i -\geo_mean(A_i x_i - b_i) + I(Cx_{N+1} - d \eq 0)
        subject to    x = A_G z
    """
    m, n = A_all[0].shape
    num_agents = len(A_all)
    x = cp.Variable((A_G.shape[0], 1), nonneg=True)
    z = cp.Variable((A_G.shape[1], 1), nonneg=True)
    f = 0
    constraints = [ x == A_G @ z,
                   C @ x[A_G.shape[0]//2:] <= d, 
                    ]
    for i in range(num_agents):
        Ai, bi = A_all[i], b_all[i]
        xi = x[i * n : (i+1) * n]
        f += -cp.geo_mean(Ai @ xi + bi)
        if eps>0:
            f += -eps * cp.sum(cp.sqrt(xi))
    prob = cp.Problem(cp.Minimize(f), constraints)
    prob.solve(solver=global_solver)
    true_lamb = constraints[0].dual_value
    true_f = f.value
    true_x = x.value
    return true_x, true_f, true_lamb


def prox_ra_query_eps_o(yi, zi, Ai, bi, rho, num_points=1, eps=1e-4, eps_sublevel=1e-1):
    # return x_i = \argmin(f_i(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    m, n = Ai.shape
    xi = cp.Variable((n, 1), nonneg=True)
    f = -cp.geo_mean(Ai @ xi + bi) 
    if eps > 0:
        f -= eps * cp.sum(cp.sqrt(xi))
    
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))))
    prob.solve(solver=global_solver)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        p_star = prob.value
        ni = cp.Parameter((n, 1))
        g = cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)), 
                          [g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8])
        ni_val = np.random.randn(n, num_points-1)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
        assert np.allclose(np.linalg.norm(ni_val, axis=0), np.ones(num_points-1))
        for t in range(num_points-1):
            ni.value = ni_val[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_ra_g_query_eps_o(yi, zi, C, d, rho, num_points=1, eps_sublevel=1e-2):
    # g(x_i) = I(Cx \leq d)
    # return x_i = \argmin(g(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    n = C.shape[1]
    xi = cp.Variable((n, 1), nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [C @ xi <= d])
    prob.solve(solver=global_solver)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        p_star = prob.value
        ni = cp.Parameter((n, 1))
        g = cp.sum(cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)),  
                      [C @ xi <= d,
                       g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8])
        ni_val = np.random.randn(n, num_points-1)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
        assert np.allclose(np.linalg.norm(ni_val, axis=0), np.ones(num_points-1))
        for t in range(num_points-1):
            ni.value = ni_val[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_ra_query_eps_conj_subgrad(yi, zi, Ai, bi, R, rho, num_points=1, eps=1e-4, eps_sublevel=1e-2):
    # return multiple points wrt price interface
    # return x_i = \argmin(f_i(x_i)) + y_i^Tx_i)
    m, n = Ai.shape
    xi = cp.Variable((n, 1), nonneg=True)
    f = -cp.geo_mean(Ai @ xi + bi) 
    if eps > 0:
        f -= eps * cp.sum(cp.sqrt(xi))
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))),
                                  [xi <= R])
    
    try:
        prob.solve(solver=global_solver)
    except:
        prob.solve(solver="CLARABEL")
    # print(prob.status)
    if num_points == 1:
        return xi.value #, prob.value
    else:    
        xs = [xi.value]    
        prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi))),
                        [xi <= R])
        try:
            prob.solve(solver=global_solver)
        except:
            prob.solve(solver="CLARABEL")
        p_star = prob.value

        ni = cp.Parameter((n, 1))
        g = cp.sum(f + cp.sum(xi.T @ yi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)), 
                          [xi <= R,
                           g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8])
        ni_val = np.random.randn(n, num_points)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
        assert np.allclose(np.linalg.norm(ni_val, axis=0), np.ones(num_points))
        for t in range(num_points-1):
            ni.value = ni_val[:, t:t+1]
            try:
                prob.solve(solver=global_solver)
            except:
                prob.solve(solver="CLARABEL", tol_gap_rel=1e-3, tol_feas=1e-3,)
            if prob.status != "optimal":
                prob.solve(solver="CLARABEL", tol_gap_rel=1e-3, tol_feas=1e-3,)
            # print(prob.status, p_star + np.abs(eps_sublevel * p_star) + 1e-8)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_ra_g_query_eps_conj_subgrad(yi, zi, C, d, rho, num_points=1, eps_sublevel=1e-2):
    # g(x_i) = I(Cx \leq d)
    # return x_i = \argmin(g(x_i)) + y_i^Tx_i)
    n = C.shape[1]
    xi = cp.Variable((n, 1), nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [C @ xi <= d])
    try:
        prob.solve(solver=global_solver)
    except:
        prob.solve(solver="CLARABEL")
    if prob.status != "optimal":
        prob.solve(solver="CLARABEL")
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        prob = cp.Problem(cp.Minimize(cp.sum(cp.sum(xi.T @ yi))), 
                      [C @ xi <= d])
        try:
            prob.solve(solver=global_solver)
        except:
            prob.solve(solver="CLARABEL")
        p_star = prob.value 

        ni = cp.Parameter((n, 1))
        g = cp.sum(cp.sum(xi.T @ yi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)),  
                      [C @ xi <= d,
                       g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8])
        ni_val = np.random.randn(n, num_points)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
        assert np.allclose(np.linalg.norm(ni_val, axis=0), np.ones(num_points))
        for t in range(num_points-1):
            ni.value = ni_val[:, t:t+1]
            try:
                prob.solve(solver=global_solver)
            except:
                prob.solve(solver="CLARABEL", tol_gap_rel=1e-3, tol_feas=1e-3,)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_ra_query_eps_o_price(yi, zi, Ai, bi, rho, num_points=1, eps=1e-4, eps_sublevel=1e-2):
    # return multiple points wrt price interface
    # return x_i = \argmin(f_i(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    # uses trust region
    m, n = Ai.shape
    xi = cp.Variable((n, 1), nonneg=True)
    f = -cp.geo_mean(Ai @ xi + bi) 
    if eps > 0:
        f -= eps * cp.sum(cp.sqrt(xi))
    
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))))
    try:
        prob.solve(solver=global_solver)
    except:
        prob.solve(solver="CLARABEL")
    # print(prob.status)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        r = np.sqrt(cp.sum_squares(xi - zi).value)
        p_star = prob.value - (rho/2) * r**2
        ni = cp.Parameter((n, 1))
        g = cp.sum(f + cp.sum(xi.T @ yi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)), 
                          [g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8,
                           cp.sum_squares(xi - zi) <= 1000*r**2])
        ni_val = np.random.randn(n, num_points)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
        assert np.allclose(np.linalg.norm(ni_val, axis=0), np.ones(num_points))
        for t in range(num_points-1):
            ni.value = ni_val[:, t:t+1]
            try:
                prob.solve(solver=global_solver)
            except:
                prob.solve(solver="CLARABEL", tol_gap_rel=1e-3, tol_feas=1e-3,)
            if prob.status != "optimal":
                prob.solve(solver="CLARABEL", tol_gap_rel=1e-3, tol_feas=1e-3,)
            # print(prob.status, p_star + np.abs(eps_sublevel * p_star) + 1e-8)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_ra_g_query_eps_o_price(yi, zi, C, d, rho, num_points=1, eps_sublevel=1e-2):
    # g(x_i) = I(Cx \leq d)
    # return x_i = \argmin(g(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    n = C.shape[1]
    xi = cp.Variable((n, 1), nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      [C @ xi <= d])
    try:
        prob.solve(solver=global_solver)
    except:
        prob.solve(solver="CLARABEL")
    if prob.status != "optimal":
        prob.solve(solver="CLARABEL")
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        p_star = prob.value - (rho/2) * cp.sum_squares(xi - zi).value
        ni = cp.Parameter((n, 1))
        g = cp.sum(cp.sum(xi.T @ yi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)),  
                      [C @ xi <= d,
                       g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8])
        ni_val = np.random.randn(n, num_points)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
        assert np.allclose(np.linalg.norm(ni_val, axis=0), np.ones(num_points))
        for t in range(num_points-1):
            ni.value = ni_val[:, t:t+1]
            try:
                prob.solve(solver=global_solver)
            except:
                prob.solve(solver="CLARABEL", tol_gap_rel=1e-3, tol_feas=1e-3,)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)
