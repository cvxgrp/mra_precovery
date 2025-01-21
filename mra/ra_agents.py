import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

from mra.utils import *
from mra.config import *





def centralized_solution_ra(A_all, b_all, C, d, R=None, eps=0):
    """
    Solve
        minimize      \sum_i f_i(x_i) = \sum_i -\geo_mean(A_i x_i - b_i)
        subject to    Cx \eq d
    """
    m, n = A_all[0].shape
    num_agents = len(A_all)
    x = cp.Variable((n * num_agents, 1), nonneg=True)
    f = 0
    constraints = [ C @ x <= d, 
                   x <= np.concatenate([d] * num_agents, axis=0) ]
    if R is not None:
        constraints += [cp.abs(x) <= R]
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


def ra_query(lamb, Ai, bi, d, eps=0):
    # return x s.t. -\lambda \in \partial f_i(x_i)) +  N(0 <= x_i <= d)
    m, n = Ai.shape
    xi = cp.Variable((n, 1), nonneg=True)
    f = -cp.geo_mean(Ai @ xi + bi) 
    if eps > 0:
        f -= eps * cp.sum(cp.sqrt(xi))
    prob = cp.Problem(cp.Minimize(cp.sum(f + xi.T @ lamb)), [xi <= d])
    try:
        prob.solve(solver=global_solver)
    except:
        print(Ai @ xi + bi)
        prob.solve(solver=cp.CLARABEL)
    # print(f"{cp.sum(f + xi.T @ lamb).value =}")
    return xi.value


def ra_query_multiple_actions(lamb, i, A, b, d, eps_sublevel=1e-2, K=1, eps=0, return_best=True):
    # return K noisy actions s.t. x is on the boundary of 
    # eps-sublevel set of (f_i(x_i) + lambda^T x_i)
    m, n = A[i].shape
    xs = []
    x_best = ra_query(lamb, A[i], b[i], d)
    p_star = cp.sum(-cp.geo_mean(A[i] @ x_best + b[i]) - eps * cp.sum(cp.sqrt(x_best)) + x_best.T @ lamb).value

    xi = cp.Variable((n, 1), nonneg=True)
    yi = cp.Parameter((n, 1))
    g = cp.sum(-cp.geo_mean(A[i] @ xi + b[i]) + xi.T @ lamb)
    if eps > 0:
        g += - eps * cp.sum(cp.sqrt(xi))
    prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ yi)), 
                      [xi <= d, 
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


def ra_query_multiple_actions_noisy_prices(lamb, i, A, b, d, percent=1e-2, K=1, eps=0):
    # introduce noise in prices proportional to each price
    # return K noisy actions (f_i(x_i) + (lambda - delta)^T x_i)
    m = lamb.size
    xs = []
    x_best = ra_query(lamb, A[i], b[i], d)

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
        f = -cp.geo_mean(A[i] @ xi + b[i])
        if eps > 0:
            f += - eps * cp.sum(cp.sqrt(xi))
        prob = cp.Problem(cp.Minimize(cp.sum(f + xi.T @ new_yi)), [xi <= d])
        for t in range(max_iter):
            new_yi.value = noisy_yi[:, t]
            prob.solve(solver=global_solver)
            xs += [xi.value]
    # n_i x K_i
    return np.concatenate(xs, axis=1)


def ra_obj_value(x, A_all, b_all, R, eps=0, idx=0):
    f = 0
    num_resources = A_all[0].shape[1]
    num_agents = len(A_all)
    for i in range(num_agents):
        Ai, bi = A_all[i], b_all[i]
        if type(x) is list:
            xi = x[i][:, idx:idx+1]
        else:
            xi = x[i*num_resources:(i+1)*num_resources][:, idx:idx+1]
        if (xi + 1e-8 < 0).any() or (np.abs(xi) - 1e-8 > R).any():
            return np.inf
        xi = np.maximum(xi, 0)
        f += - np.prod(Ai @ xi + bi)**(1./bi.size) - eps * np.sum(np.sqrt(xi))
    return f


def ra_query_xqf(lamb, Ai, bi, d, eps=0):
    m, n = Ai.shape
    x_val = ra_query(lamb, Ai, bi, d)
    xi = cp.Variable((n, 1), nonneg=True)
    f = -cp.geo_mean(Ai @ xi + bi) - eps * cp.sum(cp.sqrt(xi))
    constraints = [xi <= x_val]
    prob = cp.Problem(cp.Minimize(f), constraints)
    prob.solve(solver=global_solver)
    q = -constraints[0].dual_value
    return x_val, q, f.value


def test_ra_query(A_all, b_all, d, num_agents, num_resources, eps):
    for i in range(num_agents):
        Ai, bi = A_all[i], b_all[i]
        func_f = lambda x: -np.prod(Ai @ x + bi)**(1./bi.size)
        for _ in range(20):
            lamb = np.random.uniform(size=(num_resources, 1)) * 100
            x_a, q_a, f_a = ra_query_xqf(lamb, Ai, bi, d)
            assert np.allclose(f_a, - np.prod(Ai @ x_a + bi)**(1./bi.size) - eps * np.sum(np.sqrt(x_a)))
            test_subgradient(x_a, q_a, f_a, func_f)
            test_subgradient(x_a, -lamb, f_a, func_f)
    print("PASSED")


def subopt_ratios(xs, A_all, b_all, true_f):
    res = []
    for x_k in xs:
        loss = ra_obj_value(x_k, A_all, b_all)
        res += [np.abs((loss - true_f) / true_f)]
    return res


def plot_single_method_ra(num_resources, resources, true_lamb, prices, ):
    cmp = sns.color_palette("hls", num_resources+1)
    fig, axs = plt.subplots(2, figsize=(6, 8), dpi=150)
    plt.subplots_adjust(hspace=0.2)
    max_T = 150
    for i in range(num_resources):
        axs[0].plot(resources[i][:max_T]/d[i], color=cmp[i], lw=0.9) 
    axs[0].set_ylim(0, 4)
    axs[0].hlines(1, 0, max_T-1, color='k', ls='--', lw=0.5)
    axs[0].set_title('Total resources used')

    for i in range(num_resources):
        axs[1].hlines(true_lamb[i], 0, max_T-1, color=cmp[i], ls='--', lw=0.5)
        axs[1].plot(prices[i][:max_T], color=cmp[i], lw=0.9) 
    axs[1].set_ylim(true_lamb.min()-0.2, true_lamb.max()+0.2)
    axs[1].set_title('Price evolution')

    font_size = 8
    for ax in axs.flat:
        ax.tick_params(axis='both', labelsize=font_size)

    plt.savefig("dual_interface_ra.pdf", format='pdf', bbox_inches='tight')


def ra_query_multiple_actions_old(lamb, Ai, bi, d, eps_sublevel=1e-2, K=1, eps=0, return_best=True):
    # return K noisy actions s.t. x is on the boundary of 
    # eps-sublevel set of (f_i(x_i) + lambda^T x_i)
    m, n = Ai.shape
    xs = []
    x_best = ra_query(lamb, Ai, bi, d)
    p_star = cp.sum(-cp.geo_mean(Ai @ x_best + bi) - eps * cp.sum(cp.sqrt(x_best)) + x_best.T @ lamb).value

    xi = cp.Variable((n, 1), nonneg=True)
    yi = cp.Parameter((n, 1))
    g = cp.sum(-cp.geo_mean(Ai @ xi + bi) - eps * cp.sum(cp.sqrt(xi)) + xi.T @ lamb)
    prob = cp.Problem(cp.Maximize(cp.sum(xi.T @ yi)), 
                      [xi <= d, g <= p_star + np.abs(eps_sublevel * p_star) + 1e-8])
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


def ra_data(num_resources, num_participants, num_agents, inner_size, num_resource_per_participant, scale=2):
    A = np.concatenate([np.eye(num_resources)] * num_participants, axis=1)
    b = np.exp(np.random.normal(loc = np.log(num_participants/5), scale = scale, size = (num_resources, 1)))

    R = 3 * b.max()
    A_all = []; b_all = [] 
    for i in  range(num_agents):
        A_all += [np.random.uniform(size=(inner_size, num_resources))]
        b_all += [(num_resource_per_participant) * np.random.uniform(size=(inner_size, 1))]
    return A_all, b_all, A, b, R
    

def init_admm_ra(A, R, num_agents, num_resources, debug=False):
    N = num_agents
    primal_var_size = A.shape[1]

    primal_var_size = A.shape[1] * 2
    public_var_size = A.shape[1]

    # agent subscription to global consensus variables
    agent2consensus = [np.arange(i * num_resources, (i+1) * num_resources) for i in range(N)]
    agent2consensus += [np.arange(N*num_resources)]
    # index of global j in agent i
    global2local = [{} for _ in range(N+1)]
    consensus2agent = [[] for _ in range(public_var_size)]
    total_num_local = 2*N*num_resources
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

    assert np.allclose(E, np.concatenate([np.eye(total_num_local // 2)]*2, axis=0))

    if debug:
        EtE = np.linalg.inv(E.T @ E)
        for j in range(len(consensus2agent)):
            assert np.allclose(EtE[j, j], 1/len(consensus2agent[j]))

    return E, agent2consensus, consensus2agent, global2local, primal_var_size, public_var_size