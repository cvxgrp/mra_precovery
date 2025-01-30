import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

from mra.utils import *
from mra.config import *


"""

Methods for RA problem
    minimize      f_1(x_1) + ... + f_N(x_N) = \sum_i -\geo_mean(A_i x_i)
    subject to    x_1 + ... + x_N \leq d

using consenus ADMM
    minimize      \sum_{i=1}^N -\geo_mean(A_i x_ii) + I(Cx_{N+1} \leq d)
    subject to    x_i,j = x_{N+1},j = z_j,   for all i, j=1, ..., N
                  

"""

def centralized_solution_resal_admm(A_all, A_G, C, d, E_g=None):
    """
    Solve
        minimize      \sum_i -\geo_mean(A_i x_i - b_i) + I(Cx_{N+1} - d \eq 0)
        subject to    x = A_G z
    """
    ni = A_all[0].shape[1]; n = A_G.shape[0]
    num_agents = len(A_all)
    x = cp.Variable((A_G.shape[0], 1), nonneg=True)
    z = cp.Variable((A_G.shape[1], 1), nonneg=True)
    f = 0
    constraints = [ x == A_G @ z,
                    C @ x[n // 2:] <= d, 
                    ]
    if E_g is not None:
        constraints += [E_g @ x[n // 2:] <= 1]
    for i in range(num_agents):
        Ai = A_all[i]
        xi = x[i * ni : (i+1) * ni]
        f += -cp.geo_mean(Ai @ xi)
    prob = cp.Problem(cp.Minimize(f), constraints)
    prob.solve(solver=global_solver)
    true_lamb = constraints[0].dual_value
    true_f = f.value
    true_x = x.value
    return true_x, true_f, true_lamb


def prox_resal_query_noisy_price_percent(yi, zi, i, As, rho, num_points=1, percent=None, E_g=None):
    # return x_i = \argmin(f_i(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    # introduce noise in prices proportional to each price
    Ai = As[i]
    m, n = Ai.shape
    xi = cp.Variable((n, 1), nonneg=True)
    f = -cp.geo_mean(Ai @ xi) 
    if E_g is not None:
        constraints = [cp.sum(xi) <= 1]
    else:
        constraints = []
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))),
                      constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
    except:
        prob.solve(solver=cp.MOSEK)
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
        prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ new_yi) + (rho/2) * cp.sum_squares(xi - zi))),
                          [cp.sum(xi) <= 1])
        for t in range(num_points-1):
            new_yi.value = deltas[:, t:t+1]
            try:
                prob.solve(solver=cp.CLARABEL)
            except:
                prob.solve(solver=cp.MOSEK)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_resal_g_query_noisy_price_percent(yi, zi, C, d, E_g, rho, num_points=1, percent=None):
    # g(x_i) = I(Cx \leq d)
    # return x_i = \argmin(g(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    # introduce noise in prices proportional to each price
    n = C.shape[1]
    xi = cp.Variable((n, 1), nonneg=True)
    if E_g is not None:
        constraints = [C @ xi <= d, E_g @ xi <= 1]
    else:
        constraints = [C @ xi <= d]
    prob = cp.Problem(cp.Minimize(cp.sum(cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      constraints)
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
        prob = cp.Problem(cp.Minimize(cp.sum(cp.sum(xi.T @ new_yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      constraints)
        for t in range(num_points-1):
            new_yi.value = deltas[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_resal_query_eps_o(yi, zi, i, As, rho, num_points=1, eps_sublevel=1e-1, E_g=None):
    # return x_i = \argmin(f_i(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    Ai = As[i]
    m, n = Ai.shape
    xi = cp.Variable((n, 1), nonneg=True)
    f = -cp.geo_mean(Ai @ xi) 
    if E_g is not None:
        constraints = [cp.sum(xi) <= 1]
    else:
        constraints = []
    prob = cp.Problem(cp.Minimize(cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))),
                      constraints)
    prob.solve(solver=global_solver)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        p_star = prob.value
        ni = cp.Parameter((n, 1))
        g = cp.sum(f + cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)), 
                          constraints + [g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8])
        ni_val = np.random.randn(n, num_points-1)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
        assert np.allclose(np.linalg.norm(ni_val, axis=0), np.ones(num_points-1))
        for t in range(num_points-1):
            ni.value = ni_val[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def prox_resal_g_query_eps_o(yi, zi, C, d, E_g, rho, num_points=1, eps_sublevel=1e-2):
    # g(x_i) = I(Cx \leq d)
    # return x_i = \argmin(g(x_i)) + y_i^Tx_i + (\rho/2)\|x_i-z\|_2^2)
    n = C.shape[1]
    xi = cp.Variable((n, 1), nonneg=True)
    if E_g is not None:
        constraints = [C @ xi <= d, E_g @ xi <= 1]
    else:
        constraints = [C @ xi <= d]
    prob = cp.Problem(cp.Minimize(cp.sum(cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))), 
                      constraints)
    prob.solve(solver=global_solver)
    if num_points == 1:
        return xi.value #, prob.value
    else:
        xs = [xi.value]
        p_star = prob.value
        ni = cp.Parameter((n, 1))
        g = cp.sum(cp.sum(xi.T @ yi) + (rho/2) * cp.sum_squares(xi - zi))
        prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ ni)),  
                      constraints + [g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8])
        ni_val = np.random.randn(n, num_points-1)
        ni_val = np.divide(ni_val, np.linalg.norm(ni_val, axis=0))
        assert np.allclose(np.linalg.norm(ni_val, axis=0), np.ones(num_points-1))
        for t in range(num_points-1):
            ni.value = ni_val[:, t:t+1]
            prob.solve(solver=cp.CLARABEL)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def resal_admm_matrices(N, num_resources, A_ineq, A_all):
    public_var_size = A_ineq.shape[1]
    # agent subscription to global consensus variables
    agent2consensus = [np.arange(i * num_resources, (i+1) * num_resources) for i in range(N)]
    agent2consensus += [np.arange(N*num_resources)]
    # index of global j in agent i
    global2local = [{} for _ in range(N+1)]
    consensus2agent = [[] for _ in range(public_var_size)]
    total_num_local = 2 * N * num_resources
    for i in range(N+1):
        for idx_j, j in enumerate(agent2consensus[i]):
            consensus2agent[j] += [i] 
            global2local[i][j] = idx_j

    E = np.zeros((total_num_local, len(consensus2agent)), dtype=int) 
    count = 0
    for i in range(len(agent2consensus)):
        for gi, j in enumerate(agent2consensus[i]):
            E[count + gi, j] = 1
        count += agent2consensus[i].size
    n_i = A_all[0].shape[1]
    E_g = np.kron(np.eye(N), np.ones((1, n_i)))

    assert np.allclose(E, np.concatenate([np.eye(total_num_local // 2)]*2, axis=0))
    return E, consensus2agent, agent2consensus, E_g
