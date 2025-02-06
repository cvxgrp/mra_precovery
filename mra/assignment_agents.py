import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from mra.config import *


    

def package_agent_query(lamb, i, num_fcs, econ, integer=True, return_obj=False):
    # return x_i s.t. -\lambda \in \partial p_i(x_i))
    assert lamb.shape == (num_fcs,), print(lamb.shape, (num_fcs,))
    if integer:
        yi = cp.Variable(num_fcs, integer=True)
    else:
        yi = cp.Variable(num_fcs)
    xi = cp.Variable(num_fcs, nonneg=True)
    pi = -econ['profit_ij'][i] @ yi 
    constraints = [econ['p_cap'][i] * yi <= xi, 
                   cp.sum(yi) <= 1,
                   yi >= 0]
    prob = cp.Problem(cp.Minimize(cp.sum(pi + xi.T @ lamb)), constraints)
    try:
        prob.solve(solver=global_solver)
    except:
        try:
            prob.solve(solver=cp.MOSEK, mosek_params=global_solver_options)
        except:
            prob.solve(solver=cp.CLARABEL)
    assert prob.status == "optimal", print(prob.status)
    # print(yi.value)
    if return_obj:
        return xi.value, prob.value
    else:
        return xi.value
    

def package_agent_query_multiple_actions(lamb, i, num_fcs, econ, integer=True,  
                                         eps_sublevel=1e-1, K=1, return_best=True):
    # return x_i s.t. -\lambda \in \partial p_i(x_i))
    assert lamb.shape == (num_fcs,)
    if not integer:
        return cvx_package_agent_query_multiple_actions(lamb, i, num_fcs, econ, integer=True,
                                          eps_sublevel=eps_sublevel, K=K, return_best=return_best)
    pi = -econ['profit_ij'][i] + econ['p_cap'][i] * lamb
    min_idx = np.argmin(pi)
    p_star = min(0, pi[min_idx])
    if p_star == 0:
        return np.zeros((num_fcs, K))
    x_best = np.zeros(num_fcs)
    x_best[min_idx] = econ['p_cap'][i]

    idx = np.where(pi <= p_star + np.abs(eps_sublevel * p_star) + 1e-8)[0]
    idx_sort = np.argsort(pi[idx])
    idx = idx[idx_sort][:K]
    # (num_fcs, K)
    xs = (econ['p_cap'][i] * np.eye(num_fcs))[:, idx]
    if idx.size < K:
        xs = np.concatenate([xs, np.tile(x_best.reshape(-1, 1), (1, K - idx.size))], axis=1)
    assert xs.shape == (num_fcs, K), print(xs.shape, (num_fcs, K), np.tile(x_best.reshape(-1, 1), (1, K - idx.size)).shape)
    return xs 
    

def ncvx_best_action_package_agent(lamb, i, econ, num_fcs):
    xis = econ['p_cap'][i] * np.ones(num_fcs)
    pi = -econ['profit_ij'][i] + econ['p_cap'][i] * lamb 
    min_idx = np.argmin(pi)
    p_star = min(0, pi[min_idx])
    if p_star == 0:
        return np.zeros((num_fcs))
    x_best = np.zeros(num_fcs)
    x_best[min_idx] = econ['p_cap'][i] 
    return x_best


def package_agent_query_multiple_actions_noisy_prices(lamb, i, num_fcs, econ, integer=True, 
                                         percent=1e-1, K=1, return_best=True):
    # return x_i s.t. -\lambda \in \partial p_i(x_i))
    assert lamb.shape == (num_fcs,)
    if not integer:
        return cvx_package_agent_query_multiple_actions_noisy_prices(lamb, i, num_fcs, econ,  
                                                                     percent=percent, K=K, return_best=return_best)
    
    x_best = ncvx_best_action_package_agent(lamb, i, econ, num_fcs)
    if K == 1:
        return x_best
    else:
        xs = np.zeros((num_fcs, K))
        xs[:, 0] = x_best
        l = -percent * np.abs(lamb)[:, np.newaxis]
        u = lamb[:, np.newaxis] - l  # yi + eps * |yi|
        l += lamb[:, np.newaxis]     # yi - eps * |yi|
        assert  (lamb <= u[:,0]).all() and (lamb >= l[:,0]).all()
        noisy_yi = l + np.multiply(np.random.rand(lamb.shape[0], K), u - l)
        for t in range(1, K):
            xs[:, t] = ncvx_best_action_package_agent(noisy_yi[:, t], i, econ, num_fcs)

    return xs 


def cvx_package_agent_query_multiple_actions(lamb, i, num_fcs, econ, integer=False, 
                                          eps_sublevel=1e-1, K=1, return_best=True):
    # return x_i s.t. -\lambda \in \partial p_i(x_i))
    # return K noisy actions s.t. x is on the boundary of 
    # eps-sublevel set of (p_i(x_i) + lambda^T x_i)
    x_best, p_star = package_agent_query(lamb, i, num_fcs, econ, integer=False, return_obj=True)

    yi = cp.Variable(num_fcs)
    xi = cp.Variable(num_fcs, nonneg=True)
    pi = -econ['profit_ij'][i] @ yi 
    obj = cp.sum(pi + xi.T @ lamb)
    constraints = [econ['p_cap'][i] * yi <= xi, 
                   cp.sum(yi) <= 1,
                   yi >= 0,
                   obj <= p_star + np.abs(eps_sublevel * p_star) + 1e-8 ]
    
    gi = cp.Parameter(num_fcs)
    prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ gi)), constraints)

    gi_val = np.random.randn(num_fcs, K)
    gi_val = np.divide(gi_val, np.linalg.norm(gi_val, axis=0))

    assert np.allclose(np.linalg.norm(gi_val, axis=0), np.ones(K))
    xs = np.zeros((num_fcs, K))
    max_iter = K-1 if return_best else K
    for t in range(max_iter):
        gi.value = gi_val[:, t]
        try: prob.solve(solver=global_solver)
        except:
            try: prob.solve(solver=cp.MOSEK, mosek_params=global_solver_options)
            except:
                try: prob.solve(solver=cp.CLARABEL)
                except: prob.solve(solver='SCS')
        xs[:, t] = xi.value
    if return_best:
        xs = np.concatenate([x_best[:, np.newaxis], xs[:, :-1]], axis=1)
    return xs


def package_agent_obj(xi, i, num_fcs, econ, integer=False, return_prob=False):
    # return p_i(x_i)
    if integer:
        yi = cp.Variable(num_fcs, integer=True)
    else:
        yi = cp.Variable(num_fcs)
    pi = -econ['profit_ij'][i] @ yi 
    constraints = [econ['p_cap'][i] * yi <= xi, 
                   cp.sum(yi) <= 1,
                   yi >= 0]
    prob = cp.Problem(cp.Minimize(pi), constraints)
    try:
        prob.solve(solver=global_solver)
    except:
        try: prob.solve(solver=cp.MOSEK, mosek_params=global_solver_options)
        except: prob.solve(solver=cp.CLARABEL)
    if return_prob:
        return pi, constraints
    else:
        return pi.value, prob.status


def fc_agent_query(lamb, j, econ, return_obj=False):
    # return x_j s.t. \lambda_j \in \partial f_j(x_j))
    yj = cp.Variable(nonneg=True)
    xj = cp.Variable(nonneg=True)
    fj = econ['fc_cap_a'][j] * cp.square(cp.pos(yj - econ['fc_min_cap'][j]))
    constraints = [xj <= yj]
    prob = cp.Problem(cp.Minimize(fj - xj * lamb[j]), constraints)
    prob.solve(solver=global_solver)
    if return_obj:
        return xj.value, prob.value
    else:
        return xj.value
    

def fc_agent_query_multiple_actions(lamb, j, econ, eps_sublevel=1e-1, K=1, return_best=True):
    # return K noisy actions s.t. x is on the boundary of 
    # eps-sublevel set of (f_j(x_j) + lambda_j^T x_j)
    x_best, p_star = fc_agent_query(lamb, j, econ, return_obj=True)
    gj = cp.Parameter()
    yj = cp.Variable(nonneg=True)
    xj = cp.Variable(nonneg=True)
    fj = econ['fc_cap_a'][j] * cp.square(cp.pos(yj - econ['fc_min_cap'][j]))
    obj = fj - xj * lamb[j]
    constraints = [xj <= yj,
                   obj <= p_star + np.abs(eps_sublevel * p_star)]
    prob = cp.Problem(cp.Maximize(cp.sum(gj * xj)), constraints)
    gj_val = np.random.randn(K)
    xs = np.zeros((1, K))
    for t in range(K):
        gj.value = gj_val[t]
        prob.solve(solver=global_solver)
        xs[0, t] = xj.value
        assert prob.status == "optimal", print(prob.status)
    if return_best:
        xs[0, 0] = x_best
    # 1 x K_i
    return xs


def fc_agent_obj(xj, j, econ, return_prob=False):
    # return f_j(x_j)
    yj = cp.Variable(nonneg=True)
    fj = econ['fc_cap_a'][j] * cp.square(cp.pos(yj - econ['fc_min_cap'][j]))
    constraints = [xj <= yj]
    prob = cp.Problem(cp.Minimize(fj), constraints)
    prob.solve(solver=global_solver)
    if return_prob:
        return fj, constraints
    else:
        return fj.value


def assignment_problem_obj_val(x_k, num_packages, num_fcs, econ, integer=False):
    obj = 0
    if (x_k < 0).any():
        return np.inf
    for i in range(num_packages):
        xi = x_k[i*num_fcs : (i+1)*num_fcs, 0]
        val, status = package_agent_obj(xi, i, num_fcs, econ, integer=integer)
        if val is None:
            print(val, status)
        obj += val

    for j in range(num_fcs):
        xj = x_k[num_fcs * num_packages + j, 0]
        obj += fc_agent_obj(xj, j, econ)
    assert num_fcs * num_packages + j + 1 == x_k.size
    return obj


def assignment_problem_milp_solution(num_packages, num_fcs, econ, integer=False, solver=cp.MOSEK,  
                                     verbose=False, mipgap=0.01, timelimit=100):
    x_p = cp.Variable((num_packages, num_fcs),) 
    x_f = cp.Variable(num_fcs, nonneg=True) 

    obj = 0
    constraints = [ cp.sum(x_p, axis=0) <= x_f ] 
    for i in range(num_packages):
        xi = x_p[i]
        pi, constraints_i = package_agent_obj(xi, i, num_fcs, econ, return_prob=True, integer=integer)
        obj += pi
        constraints += constraints_i
        
    for j in range(num_fcs):
        xj = x_f[j]
        fj, constraints_j = fc_agent_obj(xj, j, econ, return_prob=True)
        obj += fj
        constraints += constraints_j

    prob = cp.Problem(cp.Minimize(obj), constraints)
    if integer:
        import gurobipy
        env = gurobipy.Env()
        env.setParam('TimeLimit', timelimit)
        env.setParam('MIPGap', mipgap)

        prob.solve(verbose=verbose, solver=cp.GUROBI, env=env)
    else:
        prob.solve(verbose=verbose, solver=solver)
    true_lamb = constraints[0].dual_value
    return x_p.value, x_f.value, obj.value, true_lamb, prob


def assignment_problem_milp_solution_lamb(lamb, num_packages, num_fcs, econ, integer=True):
    # minimize Lagrangian L(x, lamb) = f(x) + lamb^T(Ax - b) wrt x
    obj = 0
    constraints = []
    for i in range(num_packages):
        xi = cp.Variable(num_fcs) 
        pi, constraints_i = package_agent_obj(xi, i, num_fcs, econ, return_prob=True, integer=integer)
        obj += pi + lamb.T @ xi
        constraints += constraints_i
        
    for j in range(num_fcs):
        xj =cp.Variable(nonneg=True)
        fj, constraints_j = fc_agent_obj(xj, j, econ, return_prob=True)
        obj += fj - lamb[j] * xj
        constraints += constraints_j

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose=False)
    return obj.value


def assignment_problem_centralized_milp_solution(num_packages, num_fcs, econ, integer=True):
    if integer:
        y_p = cp.Variable((num_packages, num_fcs), integer=True)
    else:
        y_p = cp.Variable((num_packages, num_fcs)) 
    y_f = cp.Variable(num_fcs, nonneg=True) 
    x_p = np.diag(econ['p_cap']) @ y_p

    obj = 0
    constraints = [ cp.sum(x_p, axis=0) <= y_f,
                    cp.sum(y_p, axis=1) <= np.ones(num_packages), # send one package per package agent
                    y_p >= 0] 
    for i in range(num_packages):
        obj -= econ['profit_ij'][i] @ y_p[i]  

    for j in range(num_fcs):
        obj += econ['fc_cap_a'][j] * cp.square(cp.pos(y_f[j] - econ['fc_min_cap'][j]))

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose=False)
    true_lamb = constraints[0].dual_value
    return y_p.value, y_f.value, obj.value, true_lamb, prob


def generate_data_assignment_problem(num_packages, num_fcs):
    # Generate problem economics
    # package consumes outbound capacity
    p_cap = np.random.randint(1, np.ceil(num_packages/2/num_fcs)+1, num_packages)

    # Profit of assigning package i to FC j is econ['profit_ij'][i,j]
    econ = {'profit_ij' : np.random.rand(num_packages, num_fcs),
            'p_cap'     : p_cap}

    # Cost for FC j to generate x units of capacity
    # econ['fc_cap_a'][j]*(x-econ['fc_min_cap'][j])**2 if x>=econ['fc_min_cap'][j], otherwise, the cost is 0
    econ['fc_cap_a'] = np.random.uniform(0.8, 1.25, size=num_fcs) * econ['profit_ij'].sum() / p_cap.sum() / 30
    econ['fc_min_cap'] = np.random.randint(1, np.ceil(np.mean(p_cap))+1, num_fcs)
    # econ['fc_min_cap'] = np.ceil(np.ones(num_fcs) * np.mean(p_cap)/2)

    return econ


def cvx_package_agent_query_multiple_actions_noisy_prices(lamb, i, num_fcs, econ,
                                          percent=1e-1, K=1, return_best=True):
    # introduce noise in prices proportional to each price
    # return K noisy actions (f_i(x_i) + (lambda - delta)^T x_i) 
    zi = cp.Variable(num_fcs)
    xi = cp.Variable(num_fcs, nonneg=True)
    pi = -econ['profit_ij'][i] @ zi 
    new_yi = cp.Parameter(num_fcs) 
    constraints = [econ['p_cap'][i] * zi <= xi, 
                   cp.sum(zi) <= 1,
                   zi >= 0]
    prob = cp.Problem(cp.Minimize(cp.sum(pi + xi.T @ new_yi)), constraints)
    new_yi.value = lamb
    prob.solve(solver=cp.CLARABEL)
    if K == 1:
        return xi.value #, prob.value
    else:
        xs = np.zeros((num_fcs, K))
        xs[:, 0] = xi.value
        l = -percent * np.abs(lamb)[:, np.newaxis]
        u = lamb[:, np.newaxis] - l  # yi + eps * |yi|
        l += lamb[:, np.newaxis]     # yi - eps * |yi|
        assert  (lamb <= u[:,0]).all() and (lamb >= l[:,0]).all()
        noisy_yi = l + np.multiply(np.random.rand(lamb.shape[0], K), u - l )
        for t in range(1, K):
            new_yi.value = noisy_yi[:, t]
            prob.solve(solver=cp.CLARABEL)
            xs[:, t] = xi.value
        assert prob.status == "optimal", print(prob.status)
    # n_i x K_i
    return xs


def fc_agent_query_multiple_actions_noisy_prices(lamb, j, econ, percent=1e-1, K=1, return_best=True):
    # introduce noise in prices proportional to each price
    # return K noisy actions (f_i(x_i) + (lambda - delta)^T x_i) 
    zj = cp.Variable(nonneg=True)
    xj = cp.Variable(nonneg=True)
    fj = econ['fc_cap_a'][j] * cp.square(cp.pos(zj - econ['fc_min_cap'][j]))
    constraints = [xj <= zj]
    new_yi = cp.Parameter()
    prob = cp.Problem(cp.Minimize(fj - xj * new_yi), constraints)
    new_yi.value = lamb[j]
    prob.solve(solver=global_solver)
    if K == 1:
        return xj.value #, prob.value
    else:
        xs = [xj.value]
        l = -percent * np.abs(lamb[j])
        u = lamb[j] - l  # yi + eps * |yi|
        l += lamb[j]     # yi - eps * |yi|
        assert  (lamb[j] <= u) and (lamb[j] >= l)
        noisy_yi = l + np.multiply(np.random.rand(1, K), u - l )

        for t in range(K-1):
            new_yi.value = noisy_yi[0, t]
            prob.solve(solver=global_solver)
            xs += [xj.value]
        assert prob.status == "optimal", print(prob.status)
    # n_i x K_i
    return np.array(xs)[np.newaxis, :]


def ap_data(num_packages, num_fcs):
    econ = generate_data_assignment_problem(num_packages = num_packages, num_fcs = num_fcs)
    A = np.concatenate([np.eye(num_fcs)] * num_packages + [-np.eye(num_fcs)], axis=1)
    b = np.zeros((num_fcs, 1))
    return econ, A, b
