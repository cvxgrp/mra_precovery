import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import random

from mra.utils import *
from mra.config import *



"""
Methods for multicommodity flow problem
    minimize      - \sum_i U_i(d_i) 
    subject to    0 <= z_i <= x_i
                  Az_i + d_i(e_si - e_ti) = 0
                  x_1 + ... + x_M <= c
wrt z_i, d_i, x_i
                
using price interface
    minimize      \sum_i ( -U_i(d_i) + I(Az_i + d_i(e_si - e_ti) = 0 &  0 <= z_i <= x_i))
    subject to    x_1 + ... + x_M <= c
                  
    
f_i(x_i) =   minimize      -U_i(d_i) 
             subject to    Az_i + d_i(e_si - e_ti) = 0 
                           0 <= z_i <= x_i


f_i^*(y_i) = max_{x_i} (x_i^T y_i - f_i(x_i))
"""


def mcf_obj_value(x, params):
    num_commodities = len(params)
    num_edges = params[0]["dimension"]
    f = 0
    for i in range(num_commodities):
        if type(x) is list:
            xi = x[i][:, :1]
        else:
            xi = x[i*num_edges : (i+1)*num_edges][:, :1]
        if (xi + 1e-6 < 0).any() or (xi - 1e-6 > params[i]["upper_bound"]).any():
            return np.inf
        status, fi = mcf_obj_i(np.minimum(params[i]["upper_bound"], np.maximum(0, xi)), params[i])
        if status == "infeasible": 
            return np.inf
        else:
            f += fi.value
    return f


def mcf_obj_i(x_i, params):
    if (x_i - 1e-6 > params['upper_bound']).any():
        return "infeasible", np.inf
    # return x s.t. -\lambda \in \partial f_i(x_i)) +  N(0 <= x_i <= d)
    num_edges = params["dimension"]
    z = cp.Variable((num_edges, 1), nonneg=True)
    d_i = cp.Variable(nonneg=True)
    f = -cp.power(d_i, 0.5) * params["b"] + cp.sum(cp.sum_squares(params["incidence"] @ z + d_i * params['f']))
    constraints = [ z <= x_i + 1e-8,\
                    params["incidence"] @ z + d_i * params['f'] == 0]
    prob = cp.Problem(cp.Minimize(f), constraints)
    try:
        prob.solve(solver="ECOS")
    except:
        return "infeasible", np.inf
    return prob.status, f


def mcf_query(lamb, params):
    # return x s.t. -\lambda \in \partial f_i(x_i)) +  N(0 <= x_i <= d)
    num_edges = params["dimension"]
    x_i = cp.Variable((num_edges, 1), nonneg=True)
    z = cp.Variable((num_edges, 1), nonneg=True)
    d_i = cp.Variable(nonneg=True)
    f = -cp.power(d_i, 0.5) * params["b"] + params["v"] * cp.sum(lamb.T @ x_i) \
                                          + cp.sum_squares(params["incidence"] @ z + d_i * params['f'])
    constraints = [ z <= x_i,
                    params["incidence"] @ z + d_i * params['f'] == 0,
                    x_i <= params["upper_bound"]
                    ]
    prob = cp.Problem(cp.Minimize(f), constraints)
    prob.solve()
    return x_i.value, f.value


def mcf_query_multiple_actions(lamb, i, params, eps_sublevel=1e-2, K=1, return_best=True):
    # return K noisy actions s.t. x is on the boundary of 
    # eps-sublevel set of (f_i(x_i) + lambda^T x_i)
    n = params["dimension"]
    xs = []
    x_best, p_star = mcf_query(lamb, params)
    x_i = cp.Variable((n, 1), nonneg=True)
    yi = cp.Parameter((n, 1))
    z = cp.Variable((n, 1), nonneg=True)
    d_i = cp.Variable(nonneg=True)
    g = -cp.power(d_i, 0.5) * params["b"] + params["v"] * cp.sum(lamb.T @ x_i) \
                                          + cp.sum_squares(params["incidence"] @ z + d_i * params['f'])
    prob = cp.Problem(cp.Maximize(cp.sum(x_i.T @ yi)), 
                      [g <= p_star + np.abs(eps_sublevel * p_star) + 1e-7,
                       z <= x_i,\
                        params["incidence"] @ z + d_i * params['f'] == 0,
                        x_i <= params["upper_bound"]],
                        )
    yi_val = np.random.randn(n, K)
    yi_val = np.divide(yi_val, np.linalg.norm(yi_val, axis=0))
    assert np.allclose(np.linalg.norm(yi_val, axis=0), np.ones(K))
    max_iter = K-1 if return_best else K
    for t in range(max_iter):
        yi.value = yi_val[:, t:t+1]
        try:
            prob.solve(solver=global_solver)
        except:
            try:
                prob.solve(solver=cp.CLARABEL)
            except:
                prob.solve(solver="OSQP")
        xs += [x_i.value]
        assert prob.status == "optimal", print(prob.status, lamb.min(), np.linalg.norm(lamb), p_star)
    if return_best:
        xs = [x_best] + xs
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def mcf_query_multiple_actions_noisy_prices(lamb, i, params, percent=1e-2, K=1):
    # introduce noise in prices proportional to each price
    # return K noisy actions (f_i(x_i) + (lambda - delta)^T x_i)
    m = lamb.size
    n = params["dimension"]
    xs = []
    x_best = mcf_query(lamb, params)[0]

    if K == 1:
        return x_best
    else:
        xs = [x_best]
        yi = params["v"] * lamb
        l = -percent * np.abs(yi[:, np.newaxis])
        u = yi[:, np.newaxis] - l  # yi + eps * |yi|
        l += yi[:, np.newaxis]     # yi - eps * |yi|
        assert (l <= yi[:, np.newaxis]).all() and (yi[:, np.newaxis] <= u).all(), \
            print(np.concatenate([l, yi[:, np.newaxis], u], axis=1))
        noisy_yi = l + np.multiply(np.random.rand(m, K), u - l )
        max_iter = K-1

        new_yi = cp.Parameter((yi.size, 1))
        x_i = cp.Variable((n, 1), nonneg=True)
        z = cp.Variable((n, 1), nonneg=True)
        d_i = cp.Variable(nonneg=True)
        f = - cp.power(d_i, 0.5) * params["b"] + cp.sum_squares(params["incidence"] @ z + d_i * params['f'])
        constraints = [ z <= x_i,\
                        params["incidence"] @ z + d_i * params['f'] == 0,
                        x_i <= params["upper_bound"]
                        ]
        prob = cp.Problem(cp.Minimize(cp.sum(f + x_i.T @ new_yi)), constraints)

        for t in range(max_iter):
            new_yi.value = noisy_yi[:, t:t+1]
            prob.solve(solver=global_solver)
            xs += [x_i.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def centralized_multi_commodity_flow(params, A_ineq, b_ineq):
    """
    Returns the true solution to the distributed optimization problem
    of  Multi Commodity Flow using CVXPY and  centralised formulation
        min. \sum_i  f_i(xi)
        s.t. x1 + ... + xK = c   
    """
    K = len(params)
    constraints = []
    f = 0
    A = params[0]['incidence']
    num_vertices, num_edges = A.shape
    x = cp.Variable((K * num_edges, 1), nonneg=True)
    for i in range(K):
        xi = x[i*num_edges : (i+1)*num_edges, :]
        fi = params[i]["f"]
        d_i = cp.Variable(nonneg=True)
        zi = cp.Variable((num_edges, 1), nonneg=True)
        f += -cp.power(d_i, 0.5) * params[i]['b'] + cp.sum_squares(A @ zi  + d_i * fi)
        constraints += [ A @ zi + d_i * fi == 0, 
                         zi <= xi, 
                         xi <= params[i]["upper_bound"]
                         ]
    constraints += [A_ineq @ x <= b_ineq]
    prob = cp.Problem(cp.Minimize(f), constraints) 
    prob.solve()#solver="ECOS")
    true_lamb = constraints[-1].dual_value
    return x.value, f.value, true_lamb


def mcf_data(num_vertices, num_edges, M):
    """
    Generate parameters for resource management of Multi-Commodity flow Agent
    Arguments:
        num_vertices: int
        num_edges: int
        M: int
            number of commodities
    Returns:
        params
    """
    n = num_vertices
    # create  incidence matrix A for directed graph
    # to make graph strongly connected -- add a cycle
    A = np.zeros((num_vertices, num_edges))
    unsampled_edges = list(range(n*(n-1)))
    cur_edge = 0; global_idx = -1
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i == j: continue
            global_idx += 1
            idx = (i+1) % (num_vertices)
            if j != idx: continue
            A[i, cur_edge] =  1
            A[j, cur_edge] = -1
            unsampled_edges.remove(global_idx)
            cur_edge += 1
    assert (len(unsampled_edges) == n*(n-1) - num_vertices) and (cur_edge == num_vertices)
    chosen_edges = random.sample(unsampled_edges, num_edges - num_vertices)
    chosen_pairs = random.sample(list(range(n*(n-1))), M)
    idx = 0; cur_edge = num_vertices
    F = np.zeros((M, num_vertices))
    cur_pair = 0

    for i in range(num_vertices):
        for j in range(num_vertices):
            if i == j: continue
            if idx in chosen_edges:
                # direction of the edge is coming from (i,j) or (j,i)
                # direction = (-1)**(int(np.random.rand() > 0.5))
                A[i, cur_edge] =  1
                A[j, cur_edge] = -1
                cur_edge += 1
            if idx in chosen_pairs:
                # source  sink pair for  i-th commodity
                F[cur_pair, i] =  1
                F[cur_pair, j] = -1
                cur_pair += 1
            idx += 1
    assert idx == (n*(n-1)) and (cur_edge  == len(chosen_edges) + num_vertices)
    assert cur_pair == M and (A.sum(axis=0) == 0).all()

    # create capacities on edges
    #low  = min(100, max(10, min(num_edges)/10 ))
    cap_min = 0.2; cap_max = 2
    cap = np.random.uniform(low=cap_min, high=cap_max, size=(num_edges, 1))  
    b_min =  0.5; b_max = 1.5
    params = [0] * M

    sigma = 1. #0.5
    volumes = np.exp(np.random.randn(M, 1) * sigma)
    mean = np.exp((sigma**2) / 2)
    A_ineq = np.concatenate([volumes[i, 0] * np.eye(num_edges) for i in range(M)] , axis=1)
    b_ineq = mean * cap 

    for i in range(M):
        params[i] = {'f': F[i:i+1].T,
                    'b': np.random.uniform(low=b_min, high=b_max),
                    'dimension': num_edges, \
                    'upper_bound': b_ineq / volumes[i, 0], \
                    "incidence": A,
                    "v": volumes[i,0]}
    return params, A_ineq, b_ineq