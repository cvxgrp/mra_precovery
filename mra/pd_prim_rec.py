import numpy as np

import selector as se
from mra.set_centers_cvxpy import *
from mra.utils import *




def x_lamb_subgrad(f_subgrad, x_k, lamb_k,  A, b, rho=1):
    # return subgradient of primal-dual point
    M = np.diag((A @ x_k > b).squeeze().astype(float))
    g_lamb_k = np.maximum(A @ x_k - b, 0)
    g_xk = f_subgrad(x_k) + A.T @ M @ (lamb_k + rho * g_lamb_k)
    return g_xk, -g_lamb_k


def primal_dual_subgradient_method(f_subgrad, fun_prim_rec, fun_obj_val, K_i, A, b, num_iters=100, 
                   true_f=None, print_freq=50, eps_x_lamb=1e-28, rho=1,
                    postprocessing=1, record_volume=False, x0=None, lamb0=None):
    if lamb0 is None:
        lamb_k = np.zeros((A.shape[0], 1))
    else:
        lamb_k = lamb0 
    if x0 is None:
        x_k = np.zeros((A.shape[1], 1))
    else:
        x_k = x0
    primal_rec_x_k = np.zeros((x_k.shape))

    all_subopts_xk, all_subopts_prec = [], []
    best_to_date_viol_xk, best_to_date_viol_prec = [np.inf], [np.inf]
    lagr_xk, lagr_prec = [], []
    f_xk, f_prec = [], []
    viol_xk, viol_prec = [], []
    viol_primal_xk, viol_primal_prec = [], []
    vols = []; thetas = [-1]; dist_x_bar_x = [-1]

    prev_lamb_k = lamb_k + 0
    prev_x_k = x_k + 0

    for epoch in range(1, num_iters):
        # g_x, g_lamb = f_subgrad(x_k, lamb_k)
        g_x, g_lamb = x_lamb_subgrad(f_subgrad,  x_k, lamb_k,  A, b, rho)
        alpha_k = 1 / (epoch * (np.square(g_x).sum() + np.square(g_lamb).sum())**0.5)
        x_k = x_k - alpha_k * g_x
        lamb_k = lamb_k - alpha_k * g_lamb

        K_now = K_i if epoch % postprocessing == 0 else 1

        if K_now >= 2:
            Zs = [np.concatenate([fun_prim_rec(lamb_k[:, 0], K=K_now), x_k], axis=1)]

            if epoch % postprocessing == 0:
                # (num_agents x K_i)
                obj_lower, u_relaxed = se.cvx_relaxation_res(lamb_k, A, b.squeeze(), 1, K_now+1, 
                                                                    A.shape[0], Zs, A.shape[1])
                u_best = u_relaxed
                primal_rec_x_k[:, 0] = Zs[0] @ u_best[0]

                # vols += [vol_bounding_box(Zs)[0]]
                if record_volume:
                    vols += [vol_inscribed_ellipsoids(Zs)[0]]
                thetas += [np.percentile(get_theta_stats(u_relaxed)["p75"], 75)]

        # lagrangian = loss + (lamb_k.T @ (A @ primal_rec_x_k - b)).sum()
        f_xk += [fun_obj_val(x_k)]
        f_prec += [fun_obj_val(primal_rec_x_k)]
        loss_prec = f_prec[-1] + (lamb_k.T @ (A @ primal_rec_x_k - b)).sum()
        loss_xk = f_xk[-1] + (lamb_k.T @ (A @ x_k - b)).sum()
        if true_f is not None:
            all_subopts_prec += [np.abs((loss_prec - true_f) / true_f)]
            all_subopts_xk += [np.abs((loss_xk - true_f) / true_f)]
        lagr_xk += [loss_xk] 
        lagr_prec += [loss_prec]

        # best violation up to date
        cur_viol_xk      = np.sum(np.maximum(0, A @ x_k - b)) + (lamb_k.T @ np.abs(A @ x_k - b)).sum()
        cur_viol_prec_xk = np.sum(np.maximum(0, A @ primal_rec_x_k - b)) + (lamb_k.T @ np.abs(A @ primal_rec_x_k - b)).sum()
        # assert K_i==1 or epoch % postprocessing != 0 or cur_viol_prec_xk - 1e-4 <= cur_viol_xk, print(cur_viol_prec_xk - 1e-4 - cur_viol_xk)
        best_to_date_viol_prec += [min(best_to_date_viol_prec[-1], cur_viol_prec_xk )] 
        best_to_date_viol_xk += [min(best_to_date_viol_xk[-1], cur_viol_xk )] 
        viol_prec += [cur_viol_prec_xk] 
        viol_xk += [cur_viol_xk] 
        viol_primal_prec += [np.sum(np.maximum(0, A @ primal_rec_x_k - b))]
        viol_primal_xk += [np.sum(np.maximum(0, A @ x_k - b))]

        dist_x_bar_x += [rel_diff(x_k, primal_rec_x_k)]    
            

        pd_rel_diff = np.linalg.norm(lamb_k - prev_lamb_k) / max(1e-8, np.linalg.norm(prev_lamb_k)) + \
                      np.linalg.norm(x_k - prev_x_k) / max(1e-8, np.linalg.norm(prev_x_k))
        if epoch % print_freq == 0 or epoch == num_iters-1:
            if record_volume:
                print(f"{epoch=}, {f_prec[-1]=:.2f}, {f_xk[-1]=:.2f}, viol_pprec={viol_primal_prec[-1]:.4E}, viol_pxk={viol_primal_xk[-1]:.4E}, {pd_rel_diff=:.4E}, {vols[-1]=:.4E}, {thetas[-1]=:.4E}, {dist_x_bar_x[-1]=:.4E}")
            else:
                print(f"{epoch=}, {f_prec[-1]=:.2f}, {f_xk[-1]=:.2f}, viol_pprec={viol_primal_prec[-1]:.4E}, viol_pxk={viol_primal_xk[-1]:.4E}, {pd_rel_diff=:.4E},{thetas[-1]=:.4E}, {dist_x_bar_x[-1]=:.4E}")
            # print(f"{epoch=}, {loss_prec=:.2f}, {loss_xk=:.2f}, viol_prec={cur_viol_prec_xk:.2f}, viol_xk={cur_viol_xk:.2f}, lmb_k={np.round(lamb_k, 2).flatten()}, {lamb_rel_diff=:.4f}, {vols[-1]=}, {thetas[-1]=}")
            
        if pd_rel_diff < eps_x_lamb:
            print(f"terminate with {pd_rel_diff=}")
            break

        prev_lamb_k = lamb_k + 0
        prev_x_k = x_k + 0

    # logging
    all_results = {}
    all_results["vols"] = vols 
    all_results["thetas"] = thetas[1:]
    all_results["subopt_prec"] = all_subopts_prec
    all_results["subopt_xk"] = all_subopts_xk
    all_results["lagr_prec"] = lagr_prec
    all_results["lagr_xk"] = lagr_xk
    all_results["f_prec"] = f_prec
    all_results["f_xk"] = f_xk
    all_results["best_viol_prec"] = best_to_date_viol_prec[1:]
    all_results["best_viol_xk"] = best_to_date_viol_xk[1:]
    all_results["viol_prec"] = viol_prec 
    all_results["viol_xk"] = viol_xk
    all_results["viol_primal_prec"] = viol_primal_prec
    all_results["viol_primal_xk"] = viol_primal_xk
    all_results["dist_x_bx"] = dist_x_bar_x[1:]

    return all_results, lamb_k, x_k, primal_rec_x_k
