import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import selector as se
from mra.set_centers_cvxpy import *
from mra.utils import *


def admm_primal_recovery(fun_agents, fun_obj_val, agent2consensus, 
                         consensus2agent, global2local, E, rho=1, num_iters = 100, true_f=None,
                         print_freq=1, postprocessing=1, eps=1e-2, eps_opt=1e-2, num_points=1,
                         noisy_prices=True, rho_primal_rec=None, use_first=False, record_vol=True,
                         residuals="prim_slack", history=1):
    
    N = len(fun_agents)
    public_var_size = len(consensus2agent)
    all_results = {}
    if rho_primal_rec is None:
        rho_primal_rec = rho

    # initial dual variable and public variable
    y_k = [np.zeros((agent2consensus[i].size, 1)) for i in range(N)]
    z_k = np.zeros((public_var_size, 1))
    z_k_old = np.zeros((public_var_size, 1))

    private_var_size = sum([agent2consensus[i].size for i in range(N)])

    all_subopts_xk, bar_all_subopts_xk = [], []
    viol_xk, bar_viol_xk = [np.inf], [np.inf]
    s_ks = []; r_ks = []; s_ks_norm = []; r_ks_norm = []
    bar_s_ks = []; bar_r_ks = []; bar_s_ks_norm = []; bar_r_ks_norm = []
    bar_xis, bar_y_k, bar_z_k = [None] * 3
    thetas = []
    vols = []
    M_rats = []
    M_vols = []
    dist_z_bar_z = []; dist_x_bar_x = []
    viol_primal_xk, viol_primal_prec = [], []

    d = np.reciprocal(np.diag(E.T@E), dtype=float) # since E is selection matrix
    inv_EtE_Et = d[:, np.newaxis] * E.T

    if noisy_prices:
        deltas = []
        for i in range(N):
            dis = np.random.randn(agent2consensus[i].size, num_points) 
            dis = np.divide(dis, np.linalg.norm(dis, axis=0)[np.newaxis, :])
            dis *= np.sqrt(rho*eps_opt/2/N)
            deltas += [dis]
        deltas = np.concatenate(deltas, axis=0)

    for epoch in range(num_iters):
        if epoch % postprocessing == 0 : 
            num_points_epoch = num_points
            # if not noisy_prices and not use_first:
            if not use_first:
                num_points_epoch += 1
        else:
            deltas = None
            num_points_epoch = 1
        xis = []
        count = 0
        for i in range(N):
            if noisy_prices:
                xis += [fun_agents[i](y_k[i], z_k[agent2consensus[i]], num_points=num_points_epoch, 
                                  deltas=deltas[count:count+agent2consensus[i].size], i=i)]
            else:
                xis += [fun_agents[i](y_k[i], z_k[agent2consensus[i]], num_points=num_points_epoch, i=i)]
            count += agent2consensus[i].size
        assert count == private_var_size

        # each global variable is an average over subscribed agents
        # z_k = average_primal_vars(xis, public_var_size, global2local, consensus2agent, idx=0)
        z_k = inv_EtE_Et @ np.concatenate(xis, axis=0)[:, :1]
        
        y_k2 = admm_new_prices(xis, y_k, z_k, rho, agent2consensus, idx=0)
        # assert np.allclose(np.concatenate(y_k2, axis=0), 
        #                    np.concatenate(y_k, axis=0) + rho * (np.concatenate(xis, axis=0)[:, :1] - E @ z_k))
        y_k = y_k2

        if epoch % postprocessing == 0 and num_points >= 2:
            # (num_agents x K_i)
            if noisy_prices or use_first:
                mix_xis = xis 
            else:
                # dont use the first point
                mix_xis = [xi[:, 1:] for xi in xis]
            obj_lower, u_relaxed = se.cvx_relaxation_res_admm(mix_xis, z_k, E, num_points, rho_primal_rec)
            if residuals == "prim_slack":
                obj_lower, u_relaxed = se.cvx_relaxation_res(lamb_k, A, b, N, K_i, 
                                                                dual_var_size, Zs,  
                                                                A_eq=A_eq, b_eq=b_eq) 
            elif residuals == "prim":
                obj_lower, u_relaxed = se.cvx_relaxation(A, b, N, Zs[0].shape[1], 
                                                                dual_var_size, Zs_history,  
                                                                A_eq=A_eq, b_eq=b_eq) 
            bar_xis = []
            for i in range(N):
                bar_xis += [mix_xis[i] @ u_relaxed[i][:, np.newaxis]]
            theta_stats = get_theta_stats(u_relaxed)
            if record_vol:
                # vi, M_vi, M_rati = vol_bounding_box(mix_xis)
                vi, M_vi, M_rati = vol_inscribed_ellipsoids(mix_xis)
                vols += [vi]
                M_vols += [M_vi]
                M_rats += [M_rati]
            thetas += [np.percentile(theta_stats["p75"], 75)]
            bar_x_k = np.concatenate(bar_xis, axis=0)
            bar_z_k = inv_EtE_Et @ bar_x_k[:, :1]
            # assert np.allclose(average_primal_vars(bar_xis, public_var_size, global2local, consensus2agent, idx=0), 
            #                    inv_EtE_Et @ np.concatenate(bar_xis, axis=0)[:, :1])
            bar_y_k = admm_new_prices(bar_xis, y_k, bar_z_k, rho, agent2consensus, idx=0)     

            log_iterates_admm(fun_obj_val, E, bar_xis, bar_z_k, z_k_old, rho, agent2consensus, bar_y_k, 
                     bar_all_subopts_xk, bar_r_ks, bar_s_ks, bar_r_ks_norm, bar_s_ks_norm, bar_viol_xk, epoch, 
                     num_iters, print_freq, true_f)
            dist_z_bar_z += [rel_diff(z_k, bar_z_k)]
            dist_x_bar_x += [rel_diff(bar_x_k, np.concatenate([xi[:, :1] for xi in mix_xis], axis=0))]    
            if epoch % print_freq == 0 or epoch == num_iters-1:
                if record_vol:
                    print("theta argmax: ", theta_stats["argmax"], f"{vols[-1]=:.4E}, {thetas[-1]=:.4E}, |z-bar z|={dist_z_bar_z[-1]:.4E}, |x-bar x|={dist_x_bar_x[-1]:.4E}")
                else:
                    print("theta argmax: ", theta_stats["argmax"], f"{thetas[-1]=:.4E}, |z-bar z|={dist_z_bar_z[-1]:.4E}, |x-bar x|={dist_x_bar_x[-1]:.4E}")

        log_iterates_admm(fun_obj_val, E, xis, z_k, z_k_old, rho, agent2consensus, y_k, 
                     all_subopts_xk, r_ks, s_ks, r_ks_norm, s_ks_norm, viol_xk, epoch, 
                     num_iters, print_freq, true_f)
        
        # assert (epoch % postprocessing != 0 or num_points==1) or bar_viol_xk[-1] - 1e-6 <= viol_xk[-1], \
        #     print(epoch, bar_viol_xk[-1] - 1e-6, viol_xk[-1])

        z_k_old = z_k + 0

    # logging
    all_results = {}
    all_results["vols"] = vols 
    all_results["M_vols"] = M_vols 
    all_results["M_rats"] = M_rats 
    all_results["thetas"] = thetas
    all_results["subopt_xk"] = all_subopts_xk
    all_results["best_xk"] = viol_xk[1:]
    all_results["s_k"] = s_ks 
    all_results["r_k"] = r_ks
    all_results["s_k_norm"] = s_ks_norm 
    all_results["r_k_norm"] = r_ks_norm

    all_results["bar_subopt_xk"] = bar_all_subopts_xk
    all_results["bar_viol_xk"] = bar_viol_xk[1:]
    all_results["bar_s_k"] = bar_s_ks 
    all_results["bar_r_k"] = bar_r_ks
    all_results["bar_s_k_norm"] = bar_s_ks_norm 
    all_results["bar_r_k_norm"] = bar_r_ks_norm
    all_results["dist_z_bz"] = dist_z_bar_z 
    all_results["dist_x_bx"] = dist_x_bar_x

    return all_results, xis, z_k, bar_xis, bar_y_k, bar_z_k


def mean_std_convex_hull(xis, bar_xis):
    N = len(bar_xis)
    std = []
    for i in range(N):
        std +=[np.std(np.linalg.norm(bar_xis[i] - xis[i], axis=0) / np.linalg.norm(bar_xis[i]))]
    std = np.array(std)
    print(np.mean(std), np.median(std), np.percentile(std, 75), np.std(std),)
    return np.mean(std)


def average_primal_vars(xis, public_var_size, global2local, consensus2agent, idx=0):
    # each global variable is an average over subscribed agents
    z_k = np.zeros((public_var_size, 1))
    for j in range(public_var_size):
        z_k[j] = sum([xis[i][:, idx:idx+1][global2local[i][j]] for i in consensus2agent[j]])/len(consensus2agent[j])
    return z_k


def admm_new_prices(xis, y_k, z_k, rho, agent2consensus, idx=0):
    new_y_k = [np.zeros((agent2consensus[i].size, 1)) for i in range(len(xis))]
    for i in range(len(xis)):
        new_y_k[i] = y_k[i] + rho * (xis[i][:, idx:idx+1] - z_k[agent2consensus[i]])
    return new_y_k


def residuals_admm(E, xis, z_k, z_k_old, rho, y_k, idx=0):
    s_k = rho * np.linalg.norm(E @ (z_k - z_k_old))
    s_k_den = np.linalg.norm(np.concatenate(y_k, axis=0))
    s_k_norm = s_k / s_k_den
    r_k = np.linalg.norm(np.concatenate(xis, axis=0)[:, idx:idx+1] - E @ z_k)
    r_k_den = max(np.linalg.norm(np.concatenate(xis, axis=0)[:, idx:idx+1]),
                  np.linalg.norm(E @ z_k))
    r_k_norm = r_k / r_k_den
    return r_k, s_k, r_k_norm, s_k_norm


def log_iterates_admm(fun_obj_val, E, xis, z_k, z_k_old, rho, agent2consensus, y_k, 
                     all_subopts_xk, r_ks, s_ks, r_ks_norm, s_ks_norm, viol_xk, epoch, 
                     num_iters, print_freq, true_f=None):
    loss_xk = fun_obj_val(E @ z_k) 
    # loss_xk = fun_obj_val(xis) 
    if true_f is not None:
        all_subopts_xk += [np.abs((loss_xk - true_f) / true_f)]

    # best violation up to date
    r_k, s_k, r_k_norm, s_k_norm = residuals_admm(E, xis, z_k, z_k_old, rho, y_k, idx=0)
    cur_viol_xk = (s_k + r_k)
    s_ks_norm += [s_k_norm]
    r_ks_norm += [r_k_norm]
    s_ks += [s_k]
    r_ks += [r_k]
    viol_xk += [cur_viol_xk] #[min(viol_xk[-1], cur_viol_xk )]

    if epoch % print_freq == 0 or epoch == num_iters-1:
        print(f"{epoch=}, {loss_xk=:.2f}, {all_subopts_xk[-1]=:.2E}, {s_k=:.2E}, {r_k=:.2E}, {cur_viol_xk=:.2E}")
            
