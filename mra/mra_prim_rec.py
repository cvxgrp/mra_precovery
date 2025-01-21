import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import selector as se
from mra.set_centers_cvxpy import *
from mra.utils import *




class MRA_Primal_Recovery:
    def __init__(self, fun_agents, primal_var_size, history, res_type="primal_compl_slack",
                 A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, relaxed=True):
        self.fun_agents = fun_agents
        self.N = len(fun_agents)
        self.primal_var_size = primal_var_size
        self.history = history
        self.Zs_history = [np.array([[]]) for _ in range(self.N)]
        self.res_type = res_type
        self.A_ineq=A_ineq; self.b_ineq=b_ineq; self.A_eq=A_eq; self.b_eq=b_eq
        self.relaxed = relaxed
        

    def query(self, lamb_k, K, epoch):
        # (number of agents) x (agent size) x (number actions to choose from)
        # Zs: N x n_i x K_i
        Zs = []
        x_k = np.zeros((self.primal_var_size, 1))
        last_idx = 0
        for i in range(self.N):
            xis = self.fun_agents[i](lamb_k[:, 0], K=K, i=i)
            count = 0
            while np.isnan(xis).any():
                xis = self.fun_agents[i](lamb_k[:, 0], K=K, i=i)
                count += 1 
            Zs += [xis]
            x_k[last_idx : last_idx + xis.shape[0], 0] = xis[:, 0]
            last_idx += xis.shape[0]

            if self.history > 1:
                if epoch == 0:
                    self.Zs_history[i] = xis
                else:
                    # replace oldest xis by the newest; TODO: remove concatenate
                    # Zs_history: N x n_i x (K_i * H)
                    self.Zs_history[i] = np.concatenate([self.Zs_history[i][:, int(epoch >= self.history):], xis], axis=1)
        assert last_idx == self.primal_var_size
        return x_k, Zs


    def primal_recovery(self, lamb_k, Zs):
        # (num_agents x K_i)
        primal_rec_x_k = np.zeros((self.primal_var_size, 1))
        obj_lower, u_relaxed = se.cvx_relaxation_residuals(Zs, lamb_k, A_ineq=self.A_ineq, b_ineq=self.b_ineq, 
                                                           A_eq=self.A_eq, b_eq=self.b_eq, res_type=self.res_type)
        if self.relaxed:
            u_best = u_relaxed
        else:
            best, greed_ref_obj = se.greedy_polishing(u_relaxed, Zs, self.A_ineq, self.b_ineq, self.N, 
                                                      Zs[0].shape[1], lamb=lamb_k, num_samples=15, debug=True)
            u_best = best[1]
        last_idx = 0
        for i in range(self.N):
            primal_rec_x_k[last_idx : last_idx + Zs[i].shape[0], 0] = Zs[i] @ u_best[i]
            last_idx += Zs[i].shape[0]
        assert last_idx == self.primal_var_size
        return primal_rec_x_k



class LogMetrics:
    def __init__(self, fun_obj_val, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, true_f=None,
                 A_constraints=None, b_constraints=None):
        self.A_ineq = A_ineq; self.b_ineq=b_ineq; self.A_eq=A_eq; self.b_eq=b_eq
        self.A_constraints = A_constraints; self.b_constraints = b_constraints
        self.b_norm = np.linalg.norm(b_constraints) if np.linalg.norm(b_constraints) > 0 else 1
        all_results = {}   
        points = ["xk", "mra_xk", "paver_xk", "proj_xk"]  
        metrics = ["f", "lagr", "viol", "viol_primal", "viol_primal_compl"]  
        if true_f is not None:
            metrics += ["subopt"]
        self.true_f = true_f 
        for metric in metrics:
            for point in points:
                all_results[metric + "_" + point] = []
        all_results["prices_deltas"] = []
        self.all_results = all_results
        self.fun_obj_val = fun_obj_val

    def record(self, lamb_k=None, x_k=None, mra_xk=None, paver_xk=None, proj_xk=None):
        f_vals = {}
        metric_funcs = {"lagr": lambda x, point: f_vals[point] + (lamb_k.T @ (self.A_constraints @ x - self.b_constraints)).sum(), 
                        "viol_primal_compl": lambda x, point: primal_compl_slackness_viol(x, lamb_k, A_ineq=self.A_ineq, b_ineq=self.b_ineq, 
                                                                  A_eq=self.A_eq, b_eq=self.b_eq), 
                        "viol_primal": lambda x, point: primal_res_viol(x, A_ineq=self.A_ineq, b_ineq=self.b_ineq, 
                                                                  A_eq=self.A_eq, b_eq=self.b_eq)}
        if self.true_f is not None:
            metric_funcs["subopt"] = lambda x, point: np.abs((f_vals[point] - self.true_f)/self.true_f)

        for point, x in {"xk": x_k, "mra_xk":mra_xk, "paver_xk":paver_xk, "proj_xk":proj_xk}.items():
            if x is not None:
                f_vals[point] = self.fun_obj_val(x)
                self.all_results["f_" + point] += [f_vals[point]]
                for metric, m_func in metric_funcs.items():
                    self.all_results[metric + "_" + point] += [m_func(x, point)]
