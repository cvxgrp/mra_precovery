import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import selector as se
from mra.set_centers_cvxpy import *
from mra.utils import *
from mra.mra_prim_rec import *




def dual_proj_subgradient(fun_agents, fun_obj_val, primal_var_size, func_alpha_k, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, price_max=None, price_min=None, alpha = 1, 
                                       relaxed=True, postprocessing=1, K_i=10, num_iters = 100, true_f=None, eps_viol=1e-8,
                                       print_freq=1, eps_lamb=1e-6, res_type="primal_compl_slack", history=1):
    assert res_type in ["primal_compl_slack", "primal"]
    A_constraints, b_constraints, dual_var_size = aggregate_constraints(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq)

    mra = MRA_Primal_Recovery(fun_agents, primal_var_size, history, res_type=res_type,
                 A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, relaxed=relaxed)
    N = len(fun_agents)
    logging = LogMetrics(fun_obj_val, A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                         A_constraints=A_constraints, b_constraints=b_constraints, true_f=true_f)

    # initial polyhedron for the set containing the optimal dual variable
    if isinstance(price_max, float):
        price_max = price_max * np.ones((dual_var_size, 1))
        price_min = price_min * np.ones((dual_var_size, 1))

    prices = [(price_max +  price_min) / 2]
    lamb_k = (price_max + price_min ) / 2
    lamb_prev = np.zeros((dual_var_size, 1))
    paver_xk = np.zeros((primal_var_size, 1))

    x_k = np.zeros((primal_var_size, 1))

    A_svd = np.linalg.svd(A_constraints, full_matrices=False)
    sum_coeffs = 0
    for epoch in range(num_iters):
        # (number of agents) x (agent size) x (number actions to choose from)
        # Zs: N x n_i x K_i
        Zs = [] 
        K_now = K_i if epoch % postprocessing == 0 else 1
        x_k, Zs = mra.query(lamb_k, K_now, epoch)

        paver_xk = (sum_coeffs / (sum_coeffs + func_alpha_k(epoch))) * paver_xk \
                         + (func_alpha_k(epoch) / (sum_coeffs + func_alpha_k(epoch))) * x_k
        sum_coeffs = sum_coeffs + func_alpha_k(epoch)
        proj_xk = l2_projection_linear_eq(A_constraints, b_constraints, x_k, A_svd=A_svd)

        if epoch % postprocessing == 0:
            mra_xk = mra.primal_recovery(lamb_k, Zs)

        logging.record(lamb_k=lamb_k, x_k=x_k, mra_xk=mra_xk, paver_xk=paver_xk, proj_xk=proj_xk)
            
        if res_type == "primal_compl_slack":
            if logging.all_results['viol_primal_compl_mra_xk'][-1] - 1e-6 > logging.all_results['viol_primal_compl_xk'][-1]:
                print(f"VIOLATION: {logging.all_results['viol_primal_compl_mra_xk'][-1] - 1e-6 - logging.all_results['viol_primal_compl_xk'][-1]}=")
        elif res_type == "primal":
            if logging.all_results['viol_primal_mra_xk'][-1] - 1e-6 > logging.all_results['viol_compl_xk'][-1]:
                print(f"VIOLATION: {logging.all_results['viol_primal_compl_mra_xk'][-1] - 1e-6 - logging.all_results['viol_primal_compl_xk'][-1]}=")
        

        lamb_rel_diff = np.linalg.norm(lamb_k - lamb_prev) / np.linalg.norm(lamb_prev)
        terminate_status = epoch >= 1 and lamb_rel_diff < eps_lamb or logging.all_results["viol_primal_compl_mra_xk"][-1] < eps_viol
        logging.all_results['prices_deltas'] += [lamb_rel_diff]

        lamb_prev = lamb_k
        if epoch % print_freq == 0 or epoch == num_iters-1 or terminate_status:
            print(f"{epoch=}, f_subopt_xk={logging.all_results['subopt_xk'][-1]:.4E}, ", 
                  f"f_subopt_mra={logging.all_results['subopt_mra_xk'][-1]:.4E}, ", 
                  f"viol_xk={logging.all_results['viol_primal_compl_xk'][-1]:.4E}, viol_mra={logging.all_results['viol_primal_compl_mra_xk'][-1]:.4E}, {lamb_rel_diff=:.4E}")
        
            if epoch >= 1 and terminate_status:
                print(f"terminate with {lamb_rel_diff=}")
                break

        subgrad = b_constraints - A_constraints @ x_k
        lamb_k = lamb_k - func_alpha_k(epoch) * subgrad
        lamb_k = np.maximum(price_min, np.minimum(lamb_k, price_max))
        prices += [lamb_k]

    return logging.all_results, lamb_k, prices, x_k, paver_xk, mra_xk



def dual_subgrad_with_averaging(fun_agents, fun_obj_val, primal_var_size, func_alpha_k, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, price_max=None, price_min=None, alpha = 1, 
                                       relaxed=True, postprocessing=1, K_i=10, num_iters = 100, true_f=None, eps_viol=1e-8,
                                       print_freq=1, eps_lamb=1e-6, res_type="primal_compl_slack", history=1):
    assert res_type in ["primal_compl_slack", "primal"]
    A_constraints, b_constraints, dual_var_size = aggregate_constraints(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq)

    mra = MRA_Primal_Recovery(fun_agents, primal_var_size, history, res_type=res_type,
                 A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, relaxed=relaxed)
    N = len(fun_agents)
    logging = LogMetrics(fun_obj_val, A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                         A_constraints=A_constraints, b_constraints=b_constraints, true_f=true_f)

    # initial polyhedron for the set containing the optimal dual variable
    if isinstance(price_max, float):
        price_max = price_max * np.ones((dual_var_size, 1))
        price_min = price_min * np.ones((dual_var_size, 1))

    prices = [(price_max +  price_min) / 2]
    lamb_k = (price_max + price_min ) / 2
    lamb_prev = np.zeros((dual_var_size, 1))
    paver_xk = np.zeros((primal_var_size, 1))

    x_k = np.zeros((primal_var_size, 1))

    A_svd = np.linalg.svd(A_constraints, full_matrices=False)
    sum_coeffs = 0
    for epoch in range(num_iters):
        # (number of agents) x (agent size) x (number actions to choose from)
        # Zs: N x n_i x K_i
        Zs = [] 
        K_now = K_i if epoch % postprocessing == 0 else 1
        x_k, Zs = mra.query(lamb_k, K_now, epoch)

        paver_xk = epoch / (epoch + 1) * primal_average + 1 / (epoch + 1) * x_k
        proj_xk = l2_projection_linear_eq(A_constraints, b_constraints, x_k, A_svd=A_svd)

        if epoch % postprocessing == 0:
            mra_xk = mra.primal_recovery(lamb_k, Zs)

        logging.record(lamb_k=lamb_k, x_k=x_k, mra_xk=mra_xk, paver_xk=paver_xk, proj_xk=proj_xk)
            
        if res_type == "primal_compl_slack":
            if logging.all_results['viol_primal_compl_mra_xk'][-1] - 1e-6 > logging.all_results['viol_primal_compl_xk'][-1]:
                print(f"VIOLATION: {logging.all_results['viol_primal_compl_mra_xk'][-1] - 1e-6 - logging.all_results['viol_primal_compl_xk'][-1]}=")
        elif res_type == "primal":
            if logging.all_results['viol_primal_mra_xk'][-1] - 1e-6 > logging.all_results['viol_compl_xk'][-1]:
                print(f"VIOLATION: {logging.all_results['viol_primal_compl_mra_xk'][-1] - 1e-6 - logging.all_results['viol_primal_compl_xk'][-1]}=")

        lamb_rel_diff = np.linalg.norm(lamb_k - lamb_prev) / np.linalg.norm(lamb_prev)
        terminate_status = epoch >= 1 and lamb_rel_diff < eps_lamb or logging.all_results["viol_primal_compl_mra_xk"][-1] < eps_viol
        logging.all_results['prices_deltas'] += [lamb_rel_diff]

        lamb_prev = lamb_k
        if epoch % print_freq == 0 or epoch == num_iters-1 or terminate_status:
            print(f"{epoch=}, f_subopt_xk={logging.all_results['subopt_xk'][-1]:.4E}, ", 
                  f"f_subopt_mra={logging.all_results['subopt_mra_xk'][-1]:.4E}, ", 
                  f"viol_xk={logging.all_results['viol_primal_compl_xk'][-1]:.4E}, viol_mra={logging.all_results['viol_primal_compl_mra_xk'][-1]:.4E}, {lamb_rel_diff=:.4E}")
        
            if epoch >= 1 and terminate_status:
                print(f"terminate with {lamb_rel_diff=}")
                break

        subgrad = b_constraints - A_constraints @ x_k
        sum_coeffs = epoch / (epoch + 1) * sum_coeffs + 1 / (epoch + 1) * func_alpha_k(epoch)
        neg_subgrad = - (b_constraints - A_constraints @ primal_average)
        if A_ineq is not None:
            neg_subgrad[:b_ineq.size] = np.maximum(0, neg_subgrad[:b_ineq.size])
        lamb_k = (epoch + 1) / (epoch + 2) * lamb_k + 1 / (epoch + 2) * (neg_subgrad / sum_coeffs)
        prices += [lamb_k]

    return logging.all_results, lamb_k, prices, x_k, paver_xk, mra_xk


