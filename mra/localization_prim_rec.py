import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import selector as se
from mra.set_centers_cvxpy import *
from mra.utils import *
from mra.mra_prim_rec import *




def price_localization_primal_recovery(fun_agents, fun_obj_val, primal_var_size, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, price_max=None, price_min=None, alpha = 1, 
                                       relaxed=True, postprocessing=1, K_i=10, num_iters = 100, method="accpm_l2", true_f=None, eps_viol=1e-8,
                                       print_freq=1, eps_lamb=1e-6, res_type="primal_compl_slack", history=1, mra_milp="greedy"):
    
    assert res_type in ["primal_compl_slack", "primal"]
    A_constraints, b_constraints, dual_var_size = aggregate_constraints(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq)

    mra = MRA_Primal_Recovery(fun_agents, primal_var_size, history, res_type=res_type,
                 A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, relaxed=relaxed, mra_milp=mra_milp)
    N = len(fun_agents)
    logging = LogMetrics(fun_obj_val, A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                         A_constraints=A_constraints, b_constraints=b_constraints, true_f=true_f)

    # initial polyhedron for the set containing the optimal dual variable
    if isinstance(price_max, float):
        price_max = price_max * np.ones((dual_var_size, 1))
        price_min = price_min * np.ones((dual_var_size, 1))
    Polyhedron_A = np.concatenate([-np.eye(dual_var_size), 
                                    np.eye(dual_var_size)], axis=0)
    Polyhedron_b = np.concatenate([-price_min, price_max], axis=0)

    prices = []
    lamb_k = (price_max + price_min ) / 2
    lamb_prev = np.zeros((dual_var_size, 1))
    paver_xk = np.zeros((primal_var_size, 1))

    b_norm = (np.linalg.norm(b_ineq)**2 if b_ineq is not None else 0 + np.linalg.norm(b_eq)**2 if b_eq is not None else 0)**0.5
    rel_eps_viol = eps_viol * b_norm if b_norm > 0 else eps_viol

    A_svd = np.linalg.svd(A_constraints, full_matrices=False)
    start_idx = A_constraints.shape[0]

    for epoch in range(num_iters):
        if epoch >= 1:
            if method == "accpm_l2":
                lamb_k = cvxpy_analytic_center_l2_cone(Polyhedron_A, Polyhedron_b, alpha, start_idx=start_idx, dual_gap=1e-3, 
                                                    primal_gap=1e-3, verbose=False)
            elif method == "accpm_l2_simple":
                lamb_k = cvxpy_analytic_center_l2(Polyhedron_A, Polyhedron_b, alpha, dual_gap=1e-3, 
                                                    primal_gap=1e-3, verbose=False)
            elif method == "accpm_prox":
                lamb_k = cvxpy_analytic_center_prox(Polyhedron_A, Polyhedron_b, lamb_k, alpha, dual_gap=1e-3, 
                                                    primal_gap=1e-3, verbose=False)
            elif method == "mve_prox":
                lamb_k = mve_center_prox(Polyhedron_A, Polyhedron_b, lamb_k, alpha, dual_gap=1e-1, solver=cp.CLARABEL)[0]
            elif method == "mve":
                lamb_k = mve_center(Polyhedron_A, Polyhedron_b, dual_gap=1e-1, solver=cp.CLARABEL)[0]
        lamb_k = np.maximum(price_min, np.minimum(lamb_k, price_max))
        prices += [lamb_k]
        # (number of agents) x (agent size) x (number actions to choose from)
        # Zs: N x n_i x K_i
        Zs = [] 
        K_now = K_i if epoch % postprocessing == 0 else 1
        x_k, Zs = mra.query(lamb_k, K_now, epoch)

        proj_xk = l2_projection_linear_eq(A_constraints, b_constraints, x_k, A_svd=A_svd)

        if epoch % postprocessing == 0:
            mra_xk = mra.primal_recovery(lamb_k, Zs)

        # add new cutting planes to feasibility set of dual variable
        Polyhedron_A = np.concatenate([Polyhedron_A, (-A_constraints @ x_k + b_constraints).T], axis=0)
        Polyhedron_b = np.concatenate([Polyhedron_b, (-A_constraints @ x_k + b_constraints).T @ lamb_k], axis=0)

        paver_xk = (epoch / (epoch + 1)) * paver_xk + (1 / (epoch + 1)) * x_k[:, :1]
        logging.record(lamb_k=lamb_k, x_k=x_k, mra_xk=mra_xk, paver_xk=paver_xk, proj_xk=proj_xk)
  
        if res_type == "primal_compl_slack":
            if logging.all_results['viol_primal_compl_mra_xk'][-1] - 1e-6 > logging.all_results['viol_primal_compl_xk'][-1]:
                print(f"VIOLATION: {logging.all_results['viol_primal_compl_mra_xk'][-1] - 1e-6 - logging.all_results['viol_primal_compl_xk'][-1]}=")
        elif res_type == "primal":
            if logging.all_results['viol_primal_mra_xk'][-1] - 1e-6 > logging.all_results['viol_primal_xk'][-1]:
                print(f"VIOLATION: {logging.all_results['viol_primal_mra_xk'][-1] - 1e-6 - logging.all_results['viol_primal_xk'][-1]}=")
        
        lamb_rel_diff = np.linalg.norm(lamb_k - lamb_prev) / np.linalg.norm(lamb_prev)
        terminate_status = lamb_rel_diff < eps_lamb or logging.all_results["viol_primal_compl_mra_xk"][-1] < rel_eps_viol
        logging.all_results['prices_deltas'] += [lamb_rel_diff]

        lamb_prev = lamb_k
        if epoch % print_freq == 0 or epoch == num_iters-1 or terminate_status:
            if res_type == "primal_compl_slack":
                print(f"{epoch=}, f_subopt_xk={logging.all_results['subopt_xk'][-1]:.4E}, ", 
                    f"f_subopt_mra={logging.all_results['subopt_mra_xk'][-1]:.4E}, ", 
                    f"viol_xk={logging.all_results['viol_primal_compl_xk'][-1]:.4E}, viol_mra={logging.all_results['viol_primal_compl_mra_xk'][-1]:.4E}, {lamb_rel_diff=:.4E}")
            elif res_type == "primal":
                print(f"{epoch=}, f_subopt_xk={logging.all_results['subopt_xk'][-1]:.4E}, ", 
                    f"f_subopt_mra={logging.all_results['subopt_mra_xk'][-1]:.4E}, ", 
                    f"viol_xk={logging.all_results['viol_primal_xk'][-1]:.4E}, viol_mra={logging.all_results['viol_primal_mra_xk'][-1]:.4E}, {lamb_rel_diff=:.4E}")
        
            if terminate_status and epoch >= 1:
                print(f"terminate with {lamb_rel_diff=}")
                break

    return logging.all_results, lamb_k, prices, x_k, paver_xk, mra_xk


