import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

from mra.utils import *
from mra.config import *





def centralized_solution_resal(A_all, C, d):
    """
    Solve
        minimize      \sum_i f_i(x_i) = \sum_i -\geo_mean(A_i x_i) + I(1^T x_i <= 1)
        subject to    Cx \eq d
    """
    m, n = A_all[0].shape
    num_agents = len(A_all)
    x = cp.Variable((n * num_agents, 1), nonneg=True)
    f = 0
    constraints = [ C @ x <= d, 
                   x <= np.concatenate([d] * num_agents, axis=0) ]

    for i in range(num_agents):
        Ai = A_all[i]
        xi = x[i * n : (i+1) * n]
        f += -cp.geo_mean(Ai @ xi)
        constraints += [cp.sum(xi) <= 1]
    prob = cp.Problem(cp.Minimize(f), constraints)
    prob.solve(solver=global_solver)
    true_lamb = constraints[0].dual_value
    true_f = f.value
    true_x = x.value
    return true_x, true_f, true_lamb


def resal_query(lamb, Ai):
    # return x s.t. -\lambda \in \partial f_i(x_i)) +  N(0 <= x_i, 1^T xi <= 1)
    m, n = Ai.shape
    xi = cp.Variable((n, 1), nonneg=True)
    f = -cp.geo_mean(Ai @ xi) 
    prob = cp.Problem(cp.Minimize(cp.sum(f + xi.T @ lamb)), 
                                    [cp.sum(xi) <= 1])
    try:
        prob.solve(solver=global_solver)
    except:
        prob.solve(solver=cp.CLARABEL)
    # print(f"{cp.sum(f + xi.T @ lamb).value =}")
    return xi.value


def resal_query_multiple_actions(lamb, i, A, eps_sublevel=1e-2, K=1, return_best=True):
    # return K noisy actions s.t. x is on the boundary of 
    # eps-sublevel set of (f_i(x_i) + lambda^T x_i)
    m, n = A[i].shape
    xs = []
    x_best = resal_query(lamb, A[i])
    p_star = cp.sum(-cp.geo_mean(A[i] @ x_best) + x_best.T @ lamb).value

    xi = cp.Variable((n, 1), nonneg=True)
    yi = cp.Parameter((n, 1))
    g = cp.sum(-cp.geo_mean(A[i] @ xi) + xi.T @ lamb)
    prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ yi)), 
                      [cp.sum(xi) <= 1, 
                       g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8])
    yi_val = np.random.randn(n, K)
    yi_val = np.divide(yi_val, np.linalg.norm(yi_val, axis=0))
    assert np.allclose(np.linalg.norm(yi_val, axis=0), np.ones(K))
    max_iter = K-1 if return_best else K
    for t in range(max_iter):
        yi.value = yi_val[:, t:t+1]
        prob.solve(solver=global_solver)
        xs += [xi.value]
        assert prob.status == "optimal", print(prob.status)
    if return_best:
        xs = [x_best] + xs
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def resal_query_multiple_actions_noisy_prices(lamb, i, A, percent=1e-2, K=1):
    # introduce noise in prices proportional to each price
    # return K noisy actions (f_i(x_i) + (lambda - delta)^T x_i)
    m = lamb.size
    xs = []
    x_best = resal_query(lamb, A[i])

    if K == 1:
        return x_best
    else:
        xs = [x_best]
        l = -percent * np.abs(lamb[:, np.newaxis])
        u = lamb[:, np.newaxis] - l  # yi + eps * |yi|
        l += lamb[:, np.newaxis]     # yi - eps * |yi|
        assert (l <= lamb[:, np.newaxis]).all() and (lamb[:, np.newaxis] <= u).all(), \
            print(np.concatenate([l, lamb[:, np.newaxis], u], axis=1))
        noisy_yi = l + np.multiply(np.random.rand(m, K), u - l )
        max_iter = K-1
        xi = cp.Variable((A[i].shape[1], 1), nonneg=True)
        new_yi = cp.Parameter((lamb.shape[0]))
        f = -cp.geo_mean(A[i] @ xi) 
        prob = cp.Problem(cp.Minimize(cp.sum(f + xi.T @ new_yi)), [cp.sum(xi) <= 1])
        for t in range(max_iter):
            new_yi.value = noisy_yi[:, t]
            prob.solve(solver=global_solver)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def resal_obj_value(x, A_all, idx=0, normalized=True):
    f = 0
    num_resources = A_all[0].shape[1]
    num_agents = len(A_all)
    for i in range(num_agents):
        Ai = A_all[i]
        if type(x) is list:
            xi = x[i][:, idx:idx+1]
        else:
            xi = x[i*num_resources:(i+1)*num_resources][:, idx:idx+1]
        if  (xi + 1e-8 < 0).any() or (normalized and (np.sum(xi) - 1e-6 >= 1)):
            # print(i, (xi + 1e-8 < 0).any(), np.sum(xi) )
            return np.inf
        xi = np.maximum(xi, 0)
        f += - np.prod(Ai @ xi)**(1./Ai.shape[0])
    return f


def subopt_ratios(xs, A_all, true_f):
    res = []
    for x_k in xs:
        loss = resal_obj_value(x_k, A_all)
        res += [np.abs((loss - true_f) / true_f)]
    return res


def resal_data(num_resources, num_participants, inner_size, debug=True):
    # set resources that are more efficient (on average) are also more scarce
    A = np.concatenate([np.eye(num_resources)] * num_participants, axis=1)

    R = np.linspace(np.sqrt(num_participants), num_participants//2, num_resources)[::-1].reshape(num_resources, 1)

    intervals = np.linspace(0, 1, num_resources+1)
    diff = intervals[1:] - intervals[:-1]
    C = np.random.uniform(size=(num_participants, inner_size, num_resources))
    C = np.multiply(C, diff.reshape(1, 1, -1)) + intervals[:-1].reshape(1, 1, -1)
    C_all = np.split(C, num_participants)
    C_all = [Ci.squeeze() for Ci in C_all]
    if debug:
        for j in range(num_resources):
            assert (C[:, :, j] >= intervals[j]).all() and (C[:, :, j] <= intervals[j+1]).all()
    return C_all, A, R
    