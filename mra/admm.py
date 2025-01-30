import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import selector as se
from mra.set_centers_cvxpy import *
from mra.utils import *
from mra.mra_prim_rec import *




def admm_consensus(fun_agents, fun_obj_val, agent2consensus, 
                         consensus2agent, E, rho=1, num_iters = 100, true_f=None,
                         print_freq=1, postprocessing=1, eps_viol=1e-8, eps_res=1e-6, K_i=1,
                         res_type="primal_compl_slack", history=1, rho_primal_rec=None,
                         normalized_res=True):
    
    N = len(fun_agents)
    public_var_size = len(consensus2agent)
    private_var_size = sum([agent2consensus[i].size for i in range(N)])
    if rho_primal_rec is None:
        rho_primal_rec = rho

    d = np.reciprocal(np.diag(E.T@E), dtype=float) # since E is selection matrix
    inv_EtE_Et = d[:, np.newaxis] * E.T

    mra = MRA_Primal_Recovery(fun_agents, private_var_size, history, res_type=res_type,
                                E=E, inv_EtE_Et=inv_EtE_Et, rho=rho)
    N = len(fun_agents)
    logging = LogMetrics(fun_obj_val, E=E, true_f=true_f, inv_EtE_Et=inv_EtE_Et,
                         rho=rho)

    # initial dual variable and public variable
    y_k = np.zeros((private_var_size, 1))
    z_k = np.zeros((public_var_size, 1))
    z_k_old = np.zeros((public_var_size, 1))
    paver_xk = np.zeros((private_var_size, 1))
    for epoch in range(num_iters):
        K_now = K_i if epoch % postprocessing == 0 else 1
        x_k, Zs = mra.query_admm(y_k, z_k, K_now, epoch, agent2consensus)

        # each global variable is an average over subscribed agents
        z_k = inv_EtE_Et @ x_k
        y_k = y_k + rho * (x_k - E @ z_k)

        paver_xk = (epoch / (epoch + 1)) * paver_xk + (1 / (epoch + 1)) * x_k[:, :1]
        proj_xk = E @ z_k
        if epoch % postprocessing == 0:
            mra_xk = mra.primal_recovery_admm(z_k, Zs)

        logging.record_admm(lamb_k=y_k, x_k=x_k, mra_xk=mra_xk, proj_xk=proj_xk, 
                            paver_xk=paver_xk, z_k_old=z_k_old, normalized=normalized_res)
            
        if res_type == "primal":
            if logging.all_results['viol_primal_mra_xk'][-1] - 1e-4 > logging.all_results['viol_primal_xk'][-1]:
                print(f"VIOLATION: {logging.all_results['viol_primal_mra_xk'][-1] - 1e-6 - logging.all_results['viol_primal_xk'][-1]}, {epoch}=")

        z_k_old = z_k + 0

        rel_res = logging.all_results['viol_primal_xk'][-1] + logging.all_results['viol_primal_compl_xk'][-1]
        terminate_status =  rel_res < eps_res or logging.all_results["viol_primal_compl_mra_xk"][-1] < eps_viol
        
        if epoch % print_freq == 0 or epoch == num_iters-1 or terminate_status:
            print(f"{epoch=}, f_subopt_xk={logging.all_results['subopt_xk'][-1]:.4E}, ", 
                  f"f_subopt_mra={logging.all_results['subopt_mra_xk'][-1]:.4E}, ", 
                  f"viol_prim_xk={logging.all_results['viol_primal_xk'][-1]:.4E}, ", 
                  f"viol_prim_mra={logging.all_results['viol_primal_mra_xk'][-1]:.4E}, {rel_res=:.4E}, {rel_diff(x_k, mra_xk):.4E}")
        
            if terminate_status and epoch >= 1:
                print(f"terminate with {rel_res=}")
                break

    mra_zk = inv_EtE_Et @ mra_xk
    mra_yk = y_k + rho * (mra_xk - E @ mra_zk)
    return logging.all_results, y_k, x_k, mra_yk, mra_xk


def mean_std_convex_hull(xis, bar_xis):
    N = len(bar_xis)
    std = []
    for i in range(N):
        std +=[np.std(np.linalg.norm(bar_xis[i] - xis[i], axis=0) / np.linalg.norm(bar_xis[i]))]
    std = np.array(std)
    print(np.mean(std), np.median(std), np.percentile(std, 75), np.std(std),)
    return np.mean(std)
     
