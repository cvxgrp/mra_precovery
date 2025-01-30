import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


from mra.config import *


def rel_diff(a, b):
    return np.linalg.norm(a-b) / max(np.linalg.norm(a), np.linalg.norm(b))


def orthog_nis(n, num_points):
    A = np.random.randn(n, num_points)
    ni_val, r = np.linalg.qr(A)
    return ni_val


def minimum_vol_ellipsoid(X):
    n, N = X.shape
    L = cp.Variable((n, n))
    b = cp.Variable((n, 1))
    constraints = [cp.upper_tri(L) == 0,
                   cp.norm2(L.T @ X + b @ np.ones((1, N)), axis=0) <= np.ones(N)]
    f = -cp.sum(cp.log(cp.diag(L)))
    prob = cp.Problem(cp.Minimize(f), constraints)
    try:
        prob.solve(solver="CLARABEL", tol_gap_rel=1e-3, tol_feas=1e-3, max_iter=50)
    except:
        prob.solve(solver="CLARABEL", tol_gap_rel=1e-1, tol_feas=1e-1, max_iter=50)
    return f.value


def vol_inscribed_ellipsoids(xis):
    # N x ni x K --> n x K
    vol = 0
    vols = np.zeros(len(xis))
    for i, x in enumerate(xis):
        vols[i] = minimum_vol_ellipsoid(x)
        vol += vols[i] 
    return vol, np.median(vols), 0


def vol_inscribed_ellipsoids_X(xis):
    # N x ni x K --> n x K
    vol = minimum_vol_ellipsoid(np.concatenate(xis, axis=0))
    return vol, vol/len(xis), 0


def vol_bounding_box(xis):
    # N x ni x K --> n x K
    vol = 0
    vols = np.zeros(len(xis))
    rats = np.zeros(len(xis))
    for i, x in enumerate(xis):
        l = x.min(axis=1)
        u = x.max(axis=1)
        vols[i] = np.mean(np.log(u - l))
        vol += vols[i] 
        rats[i] = (u-l).max() / (u-l).min()
    return vol, np.median(vols), np.median(rats)


def get_theta_stats(u_relaxed):
    N = len(u_relaxed)
    theta_stats = {"max":np.zeros(N), "std":np.zeros(N), "m":np.zeros(N), 
                   "M":np.zeros(N), "p75":np.zeros(N), "argmax":np.zeros(N, int)}
    for i in range(N):
        theta_stats["max"][i] = np.max(u_relaxed[i])
        theta_stats["M"][i] = np.median(u_relaxed[i])
        theta_stats["p75"][i] = np.percentile(u_relaxed[i], 75)
        theta_stats["std"][i] = np.std(u_relaxed[i])
        theta_stats["m"][i] = np.mean(u_relaxed[i])
        theta_stats["argmax"][i] = np.argmax(u_relaxed[i])
    return theta_stats


def primal_violations(C, d, xs):
    res = []
    for x_k in xs:
        res += [np.linalg.norm(np.maximum(0, C @ x_k - d))]
    return res


def residuals_admm(E, xis, z_k, z_k_old, rho, y_k):
    s_k = rho * np.linalg.norm(E @ (z_k - z_k_old))
    s_k_den = np.linalg.norm(np.concatenate(y_k, axis=0))
    s_k_norm = s_k / s_k_den
    r_k = np.linalg.norm(np.concatenate(xis, axis=0)[:, :1] - E @ z_k)
    r_k_den = max(np.linalg.norm(np.concatenate(xis, axis=0)[:, :1]),
                  np.linalg.norm(E @ z_k))
    r_k_norm = r_k / r_k_den
    return r_k, s_k, r_k_norm, s_k_norm


def cvxpy_max_viol_mean(C, d, X, f="max"):
    num = X.shape[1]
    theta = cp.Variable((num, 1), nonneg=True)
    x = X @ theta
    if f == "max":
        obj = cp.max(cp.pos(C @ x - d))
    else:
        obj = cp.norm(cp.pos(C @ x - d), 2)
    constraints = [cp.sum(theta) == 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    return X @ theta.value


def primal_average(all_xs, mode, C, d):
    def mean(max_iter):
        res = [all_xs[0]]
        for i in range(1, max_iter):
            res += [(res[-1]*i + all_xs[i]) / (i + 1)]
            assert np.allclose(res[-1], sum(all_xs[: i + 1]) / (i + 1))
        return res
    if mode == "mean":
        return mean(len(all_xs))
    elif "mean_" in mode:
        ln = int(mode.split("_")[1])
        res = mean(ln)
        for i in range(ln, len(all_xs)):
            res += [(res[-1] * ln - all_xs[i - ln] + all_xs[i]) / ln]
            assert np.allclose(res[-1], sum(all_xs[i - ln + 1 : i + 1]) / ln) and ln == len(all_xs[i-ln+1 : i+1])
        return res
    elif "viol" in mode:
        res = [all_xs[0]]
        num = int(mode.split("_")[1])
        for i in range(1, len(all_xs)):
            X = np.concatenate(all_xs[max(0, i - num + 1) : i+1], axis=1)
            assert X.shape[1] <= num, print(X.shape, num)
            res += [cvxpy_max_viol_mean(C, d, X, mode[:3])]
        return res
    elif mode == "normal":
        return all_xs
    


def test_subgradient(x_a, q_a, f_a, func_f, func_data=None, num_iters=10, eps=1e-7):
    if func_data is  None:
        func_data = lambda: np.random.uniform(size=x_a.shape) * 100
    for _ in range(num_iters):
        b = func_data()
        f_b = func_f(b)
        assert f_b  + eps >= f_a + q_a.T.dot(b - x_a), print(f_b  + eps, f_a + q_a.T.dot(b - x_a))


def l2_projection_linear_eq(A, b, x, A_svd=None):
    # \ell_2 projection of x onto Ax=b
    if A_svd is None:
        U, S, _ = np.linalg.svd(A, full_matrices=False)
    else:
        U, S, _ = A_svd
    S2_inv = np.reciprocal(S**2, where=S!=0)
    return x - A.T @ ((U * S2_inv) @ (U.T @(A@x - b)))


def primal_compl_slackness_viol(x_k, lamb_k, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None):
    res = 0
    if A_ineq is not None:
        res += np.sum(np.maximum(0, A_ineq @ x_k - b_ineq)) + (lamb_k[:A_ineq.shape[0]].T @ np.abs(A_ineq @ x_k - b_ineq)).sum()
    if A_eq is not None:
        res += np.sum(np.abs(A_eq @ x_k - b_eq)) 
    return res


def primal_res_viol(x_k, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None):
    res = 0
    if A_ineq is not None:
        res += np.sum(np.maximum(0, A_ineq @ x_k - b_ineq)) 
    if A_eq is not None:
        res += np.sum(np.abs(A_eq @ x_k - b_eq)) 
    return res

def primal_residuals_admm(E, xis, z_k, z_k_old, rho, y_k, normalized=True):
    if isinstance(xis, list):
        x_k = np.concatenate(xis, axis=0)[:, :1]
    else:
        x_k = xis
    r_k = np.linalg.norm(x_k - E @ z_k)
    r_k_den = max(np.linalg.norm(x_k), np.linalg.norm(E @ z_k))
    r_k_norm = r_k / r_k_den
    if normalized: return r_k_norm
    else:  return r_k

def dual_residuals_admm(E, xis, z_k, z_k_old, rho, y_k, normalized=True):
    s_k = rho * np.linalg.norm(E @ (z_k - z_k_old))
    s_k_den = np.linalg.norm(np.concatenate(y_k, axis=0))
    s_k_norm = s_k / s_k_den
    if normalized: return s_k_norm
    else:  return s_k

def primal_dual_residuals_admm(E, xis, z_k, z_k_old, rho, y_k, normalized=True):
    if isinstance(xis, list):
        x_k = np.concatenate(xis, axis=0)[:, :1]
    else:
        x_k = xis
    r_k = np.linalg.norm(x_k - E @ z_k)
    r_k_den = max(np.linalg.norm(x_k), np.linalg.norm(E @ z_k))
    r_k_norm = r_k / r_k_den
    s_k = rho * np.linalg.norm(E @ (z_k - z_k_old))
    s_k_den = np.linalg.norm(np.concatenate(y_k, axis=0))
    s_k_norm = s_k / s_k_den
    if normalized: return r_k_norm + s_k_norm
    else:  return r_k + s_k

def aggregate_constraints(A_ineq=None, b_ineq=None, A_eq=None, b_eq=None):
    dual_var_size = 0
    A_constraints, b_constraints = [], []
    if A_ineq is not None:
        dual_var_size += b_ineq.size
        A_constraints += [A_ineq]
        b_constraints += [b_ineq]
    if A_eq is not None:
        dual_var_size += b_eq.size
        A_constraints += [A_eq]
        b_constraints += [b_eq]
    A_constraints = np.concatenate(A_constraints, axis=0)
    b_constraints = np.concatenate(b_constraints, axis=0)
    # b_norm = np.linalg.norm(b_constraints) if np.linalg.norm(b_constraints) > 0 else 1
    return A_constraints, b_constraints, dual_var_size


def plot_func_subopt_all(all_res, all_results_eps, all_results_noisy_y, true_f, eps_sublevel, percent, 
                         filename=None, T=None, folder="../plots/"):
    if T is None:
        T = len(all_res["f_xk"])
    
    cmp = ["orange", "red", "blue", "forestgreen", "violet"]
    fig, axs = plt.subplots(1, figsize=(6, 4), dpi=120)

    axs.plot((np.abs(np.array(all_res["f_xk"]) - true_f))[:T]/np.abs(true_f), 
                color=cmp[0], label=r"$x^k$", alpha=0.9, lw=0.5, marker='.', markersize=2) 
    axs.plot((np.abs(np.array(all_results_eps["f_mra_xk"])[:T] - true_f))/np.abs(true_f), 
                color=cmp[1], label=rf"$\bar x^k, ~\epsilon_v={int(eps_sublevel*100)}\%$", 
                alpha=0.9, lw=0.5, marker='.', markersize=2) 
    axs.plot((np.abs(np.array(all_results_noisy_y["f_mra_xk"])[:T] - true_f))/np.abs(true_f), 
                color=cmp[2], label=rf"$\bar x^k, ~\epsilon_p={int(percent*100)}\%$", 
                alpha=0.9, lw=0.5, marker='.', markersize=2) 
    axs.plot((np.abs(np.array(all_res["f_paver_xk"])[:T] - true_f))/np.abs(true_f), 
                color=cmp[3], label=r"$\frac{1}{k}\sum_{j=1}^k x^j$", 
                alpha=0.9, lw=0.5, marker='.', markersize=2) 
    axs.plot((np.abs(np.array(all_res["f_proj_xk"])[:T] - true_f))/np.abs(true_f), 
                color=cmp[4], label=r"$\Pi(x^k)$", 
                alpha=0.9, lw=0.5, marker='.', markersize=2) 

    print(f"{T=}")
    print((np.abs(np.array(all_res["f_xk"])[:T] - true_f)/np.abs(true_f))[-1],
        "sub_eps", (np.abs(np.array(all_results_eps["f_mra_xk"])[:T] - true_f)/np.abs(true_f))[-1],
        "noisy_y", (np.abs(np.array(all_results_noisy_y["f_mra_xk"])[:T] - true_f)/np.abs(true_f))[-1],
        "proj", ((np.abs(np.array(all_res["f_proj_xk"])[:T] - true_f))/np.abs(true_f)).min())
    axs.legend()
    axs.set_ylabel(r"$|f(x) - f^\star|/|f^\star|$")
    axs.set_xlabel(r'$k$')
    axs.set_yscale('log')
    if filename is not None:
        plt.savefig(folder + "%s_func_subopt.pdf"%filename)


def plot_lamb_k_diff(all_res, filename=None, folder="../plots/"):
    cmp = ["orange", "red", "blue", "forestgreen", "violet", "cyan"]
    fig, axs = plt.subplots(1, figsize=(6, 4), dpi=120)

    axs.plot(np.array(all_res["prices_deltas"]), 
                color=cmp[-1], alpha=0.9, lw=0.5, marker='.', markersize=2)  

    axs.set_ylabel(r'$\|\lambda_k - \lambda_{k-1} \|_2/\|\lambda_{k-1}\|_2$')
    axs.set_xlabel(r'$k$')
    axs.set_yscale('log')
    if filename is not None:
        plt.savefig(folder + "%s_prices_deltas.pdf"%filename)


def plot_prim_complem_residuals(all_res, all_results_eps, all_results_noisy_y, b_norm0, eps_sublevel, percent, 
                                filename=None, T=None, folder="../plots/", admm=False):
    if T is None:
        T = len(all_res["f_xk"])
    cmp = ["orange", "red", "blue", "forestgreen", "violet"]
    fig, axs = plt.subplots(1, figsize=(6, 4), dpi=120)
    b_norm = 1 if b_norm0 == 0 else b_norm0

    axs.plot(np.array(all_res["viol_primal_compl_xk"])[:T] / b_norm,  
                color=cmp[0], label=r"$x^k$", alpha=0.9, lw=0.5, marker='.', markersize=2)  
    axs.plot(np.array(all_results_eps["viol_primal_compl_mra_xk"])[:T] / b_norm, 
                color=cmp[1], label=rf"$\bar x^k, ~\epsilon_v={int(eps_sublevel*100)}\%$", 
                alpha=0.9, lw=0.5, marker='.', markersize=2) 
    axs.plot(np.array(all_results_noisy_y["viol_primal_compl_mra_xk"])[:T] / b_norm,
                color=cmp[2], label=rf"$\bar x^k, ~\epsilon_p={int(percent*100)}\%$", 
                alpha=0.9, lw=0.5, marker='.', markersize=2) 
    axs.legend()

    print(f"{T=}")
    print(np.array(all_res["viol_primal_xk"])[:T][-1] / b_norm,
        "sub_eps", np.array(all_results_eps["viol_primal_mra_xk"])[:T][-1] / b_norm,
        "noisy_y", np.array(all_results_noisy_y["viol_primal_mra_xk"])[:T][-1] / b_norm)

    if admm:
        axs.set_ylabel(r'$r_p+r_d$')
    else:
        if b_norm0 > 0:
            axs.set_ylabel(r'$(r_p+r_c)/\|b\|$')
        else:
            axs.set_ylabel(r'$r_p+r_c$')
    axs.set_xlabel(r'$k$')
    axs.set_yscale('log')

    if filename is not None:
        plt.savefig(folder + "%s_prim_slack_resid.pdf"%filename)


def plot_prim_residuals(all_res, all_results_eps, all_results_noisy_y, b_norm0, eps_sublevel, percent, 
                        filename=None, T=None, folder="../plots/", admm=False):
    if T is None:
        T = len(all_res["f_xk"])
    cmp = ["orange", "red", "blue", "forestgreen", "violet"]
    fig, axs = plt.subplots(1, figsize=(6, 4), dpi=120)
    b_norm = 1 if b_norm0 == 0 else b_norm0

    axs.plot(np.array(all_res["viol_primal_xk"])[:T]/ b_norm, 
                color=cmp[0], label=r"$x^k$", alpha=0.9, lw=0.5, marker='.', markersize=2)  
    axs.plot(np.array(all_results_eps["viol_primal_mra_xk"])[:T] / b_norm, 
                color=cmp[1], label=rf"$\bar x^k, ~\epsilon_v={int(eps_sublevel*100)}\%$", alpha=0.9, lw=0.5, marker='.', markersize=2) 
    axs.plot(np.array(all_results_noisy_y["viol_primal_mra_xk"])[:T] / b_norm,
                color=cmp[2], label=rf"$\bar x^k, ~\epsilon_p={int(percent*100)}\%$", alpha=0.9, lw=0.5, marker='.', markersize=2) 
    axs.plot(np.array(all_res["viol_primal_paver_xk"])[:T] / b_norm, 
                color=cmp[3], label=r"$\frac{1}{k}\sum_{j=1}^k x^j$", alpha=0.9, lw=0.5, marker='.', markersize=2) 

    axs.legend()

    print(f"{T=}")
    print(np.array(all_res["viol_primal_xk"])[:T][-1] / b_norm,
        "sub_eps", np.array(all_results_eps["viol_primal_mra_xk"])[:T][-1] / b_norm,
        "noisy_y", np.array(all_results_noisy_y["viol_primal_mra_xk"])[:T][-1] / b_norm)


    if b_norm0 > 0 and not admm:
        axs.set_ylabel(r'$r_p/\|b\|$')
    else:
        axs.set_ylabel(r'$r_p$')
    axs.set_xlabel(r'$k$')
    axs.set_yscale('log')

    if filename is not None:
        plt.savefig(folder + "%s_primal_resid.pdf"%filename)


def plot_lagr_subopt_all(all_res, all_results_eps, all_results_noisy_y, eps_sublevel, percent, 
                         true_f=None, filename=None, T=None, folder="../plots/"):
    if T is None:
        T = len(all_res["f_xk"])
    print(f"{T=}")
    cmp = ["orange", "red", "blue", "forestgreen", "violet"]
    fig, axs = plt.subplots(1, figsize=(6, 4), dpi=120)

    axs.plot(np.abs((np.array(all_res["lagr_xk"])[:T] - true_f) / true_f), 
                color=cmp[0], label=r"$x^k$", alpha=0.9, lw=0.5, marker='.', markersize=2)  
    axs.plot(np.abs((np.array(all_results_eps["lagr_mra_xk"][:T]) - true_f) / true_f), 
                color=cmp[1], label=rf"$\bar x^k, ~\epsilon_v={int(eps_sublevel*100)}\%$", alpha=0.9, lw=0.5, marker='.', markersize=2) 
    axs.plot(np.abs((np.array(all_results_noisy_y["lagr_mra_xk"][:T]) - true_f) / true_f),  
                color=cmp[2], label=rf"$\bar x^k, ~\epsilon_p={int(percent*100)}\%$", alpha=0.9, lw=0.5, marker='.', markersize=2) 

    axs.legend()
    
    axs.set_ylabel(r'$|(\mathcal{L}(x^k, y^k) - f^\star) / f^\star|$')
    axs.set_xlabel(r'$k$')
    axs.set_yscale('log')
    if filename is not None:
        plt.savefig(folder + "%s_lagr_subopt.pdf"%filename)


def plot_eps_all_metrics_3x(all_results_eps, all_results_noisy_y, K_i, true_f, 
                                b_norm0, percents, filename=None, T=None, folder="../plots/", admm=False):
    cmp = ["red", "tomato", "blue", "royalblue"]
    lstyle = ["-", "--"]
    b_norm = 1 if b_norm0 == 0 else b_norm0
    mosaic = """
    AABB
    CCDD
    """
    fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=(12, 10), sharex=False, sharey=False)
    for i, eps in enumerate(percents):
        ax_dict['A'].plot((np.abs(np.array(all_results_eps[eps]["f_mra_xk"]) - true_f)) / np.abs(true_f),
                        color=cmp[i], label=rf"$\epsilon_v={int(eps * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
        ax_dict['A'].plot((np.abs(np.array(all_results_noisy_y[eps]["f_mra_xk"]) - true_f)) / np.abs(true_f),
                        color=cmp[2 + i], label=rf"$\epsilon_p={int(eps * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
    ax_dict['A'].set_ylabel(r"$|f(x) - f^\star|/|f^\star|$", fontsize=14)
    ax_dict['A'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['A'].set_yscale('log')

    for i, eps in enumerate(percents):
        ax_dict['B'].plot(np.array(all_results_eps[eps]["viol_primal_mra_xk"]) / b_norm,
                        color=cmp[i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
        ax_dict['B'].plot(np.array(all_results_noisy_y[eps]["viol_primal_mra_xk"]) / b_norm,
                        color=cmp[2 + i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
    if b_norm0 > 0 and not admm:
        ax_dict['C'].set_ylabel(r'$r_p/\|b\|$')
    else:
        ax_dict['C'].set_ylabel(r'$r_p$')
    ax_dict['B'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['B'].set_yscale('log')

    for i, eps in enumerate(percents):
        ax_dict['C'].plot(np.array(all_results_eps[eps]["viol_primal_compl_mra_xk"]) / b_norm,
                        color=cmp[i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
        ax_dict['C'].plot(np.array(all_results_noisy_y[eps]["viol_primal_compl_mra_xk"]) / b_norm,
                        color=cmp[2 + i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
    
    if admm:
        ax_dict['C'].set_ylabel(r'$r_p+r_d$')
    else:
        if b_norm0 > 0:
            ax_dict['C'].set_ylabel(r'$(r_p+r_c)/\|b\|$', fontsize=14)
        else:
            ax_dict['C'].set_ylabel(r'$(r_p+r_c)$', fontsize=14)
    ax_dict['C'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['C'].set_yscale('log')


    for i, eps in enumerate(percents):
        ax_dict['D'].plot(subopt_of_best_feas_point(all_results_eps[eps], true_f, b_norm),
                        color=cmp[i], label=rf"$\epsilon_v={int(eps * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
        ax_dict['D'].plot(subopt_of_best_feas_point(all_results_noisy_y[eps], true_f, b_norm),
                        color=cmp[2 + i], label=rf"$\epsilon_p={int(eps * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
    ax_dict['D'].set_ylabel(r"$|f(x) - f^\star|/|f^\star|$", fontsize=14)
    ax_dict['D'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['D'].set_yscale('log')
    handles, labels = ax_dict['A'].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if filename is not None:
        plt.savefig(folder + "%s_all_metrics_square.pdf" % filename, bbox_inches="tight")    


def plot_N_all_metrics_3x(all_results_eps, all_results_noisy_y, eps_sublevel, true_f, 
                                b_norm0, num_points, filename=None, T=None, folder="../plots/", admm=False):
    cmp = ["red", "tomato", "blue", "royalblue"]
    lstyle = ["-", "--"]
    b_norm = 1 if b_norm0 == 0 else b_norm0
    mosaic = """
    AABB
    .CC.
    """
    fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=(12, 10), sharex=False, sharey=False)

    for i, K_i in enumerate(num_points):
        ax_dict['A'].plot((np.abs(np.array(all_results_eps[K_i]["f_mra_xk"]) - true_f)) / np.abs(true_f),
                        color=cmp[i], label=rf"$\epsilon_v={int(eps_sublevel * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
        ax_dict['A'].plot((np.abs(np.array(all_results_noisy_y[K_i]["f_mra_xk"]) - true_f)) / np.abs(true_f),
                        color=cmp[2 + i], label=rf"$\epsilon_p={int(eps_sublevel * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
    ax_dict['A'].set_ylabel(r"$|f(x) - f^\star|/|f^\star|$", fontsize=14)
    ax_dict['A'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['A'].set_yscale('log')

    for i, K_i in enumerate(num_points):
        ax_dict['B'].plot(np.array(all_results_eps[K_i]["viol_primal_compl_mra_xk"]) / b_norm,
                        color=cmp[i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
        ax_dict['B'].plot(np.array(all_results_noisy_y[K_i]["viol_primal_compl_mra_xk"]) / b_norm,
                        color=cmp[2 + i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
    if admm:
        ax_dict['B'].set_ylabel(r'$r_p+r_d$')
    else:
        if b_norm0 > 0:
            ax_dict['B'].set_ylabel(r'$(r_p+r_c)/\|b\|$', fontsize=14)
        else:
            ax_dict['B'].set_ylabel(r'$(r_p+r_c)$', fontsize=14)
    ax_dict['B'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['B'].set_yscale('log')

    for i, K_i in enumerate(num_points):
        ax_dict['C'].plot(np.array(all_results_eps[K_i]["viol_primal_mra_xk"]) / b_norm,
                        color=cmp[i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
        ax_dict['C'].plot(np.array(all_results_noisy_y[K_i]["viol_primal_mra_xk"]) / b_norm,
                        color=cmp[2 + i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
    
    if b_norm0 > 0 and not admm:
        ax_dict['C'].set_ylabel(r'$r_p/\|b\|$')
    else:
        ax_dict['C'].set_ylabel(r'$r_p$')
    ax_dict['C'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['C'].set_yscale('log')

    handles, labels = ax_dict['A'].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if filename is not None:
        plt.savefig(folder + "%s_all_metrics_square.pdf" % filename, bbox_inches="tight")


def subopt_of_best_feas_point(all_points, true_f, b_norm, eps_feas = 1e-6):
    fs, rs = all_points["f_mra_xk"], all_points["viol_primal_mra_xk"]
    res = [np.inf]
    best_res = np.inf
    for k in range(len(fs)):
        subopt_k = np.abs(fs[k] - true_f) / np.abs(true_f)
        resid_k = rs[k] / b_norm
        if resid_k < eps_feas:
            res += [min(res[-1], subopt_k)]
        elif best_res < eps_feas:
            res += [res[-1]]
            continue
        elif resid_k < best_res:
            res += [subopt_k]
        else:
            res += [res[-1]] 
        best_res = min(best_res, resid_k)
    return res[1:]


def subopt_of_best_feas_point_only(all_points, true_f, b_norm, eps_feas = 1e-6):
    fs, rs = all_points["f_mra_xk"], all_points["viol_primal_mra_xk"]
    return subopt_of_best_feas_point_base(fs, rs, true_f, b_norm, eps_feas = 1e-6)


def subopt_of_best_feas_point_base(fs, rs, true_f, b_norm, eps_feas=1e-6):
    res = [np.inf]
    best_res = np.inf
    for k in range(len(fs)):
        subopt_k = np.abs(fs[k] - true_f) / np.abs(true_f)
        resid_k = rs[k] / b_norm
        if resid_k < eps_feas:
            res += [min(res[-1], subopt_k)]
        elif best_res < eps_feas:
            res += [res[-1]]
            continue
        else:
            res += [np.inf] 
        best_res = min(best_res, resid_k)
    return res[1:]


def plot_N_all_metrics_4x(all_results_eps, all_results_noisy_y, eps_sublevel, true_f, 
                                b_norm0, num_points, filename=None, T=None, folder="../plots/"):
    cmp = ["red", "tomato", "blue", "royalblue"]
    lstyle = ["-", "--"]
    b_norm = 1 if b_norm0 == 0 else b_norm0
    mosaic = """
    AABB
    CCDD
    """

    fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=(12, 10), sharex=False, sharey=False)

    for i, K_i in enumerate(num_points):
        ax_dict['A'].plot((np.abs(np.array(all_results_eps[K_i]["f_mra_xk"]) - true_f)) / np.abs(true_f),
                        color=cmp[i], label=rf"$\epsilon_v={int(eps_sublevel * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=1)
        ax_dict['A'].plot((np.abs(np.array(all_results_noisy_y[K_i]["f_mra_xk"]) - true_f)) / np.abs(true_f),
                        color=cmp[2 + i], label=rf"$\epsilon_p={int(eps_sublevel * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=1)
    ax_dict['A'].set_ylabel(r"$|f(x) - f^\star|/|f^\star|$", fontsize=14)
    ax_dict['A'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['A'].set_yscale('log')

    for i, K_i in enumerate(num_points):
        ax_dict['B'].plot(np.array(all_results_eps[K_i]["viol_primal_mra_xk"]) / b_norm,
                        color=cmp[i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=1)
        ax_dict['B'].plot(np.array(all_results_noisy_y[K_i]["viol_primal_mra_xk"]) / b_norm,
                        color=cmp[2 + i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=1)
    if b_norm0 > 0:
        ax_dict['B'].set_ylabel(r'$r_p/\|b\|$', fontsize=14)
    else:
        ax_dict['B'].set_ylabel(r'$r_p$', fontsize=14)
    ax_dict['B'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['B'].set_yscale('log')


    for i, K_i in enumerate(num_points):
        ax_dict['C'].plot(np.array(all_results_eps[K_i]["viol_primal_compl_mra_xk"]) / b_norm,
                        color=cmp[i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=1)
        ax_dict['C'].plot(np.array(all_results_noisy_y[K_i]["viol_primal_compl_mra_xk"]) / b_norm,
                        color=cmp[2 + i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=1)
    if b_norm0 > 0:
        ax_dict['C'].set_ylabel(r'$(r_p+r_c)/\|b\|$', fontsize=14)
    else:
        ax_dict['C'].set_ylabel(r'$(r_p+r_c)$', fontsize=14)
    ax_dict['C'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['C'].set_yscale('log')

    eps_subopt = (np.abs(np.array(all_results_eps[K_i]["f_mra_xk"]) - true_f)) / np.abs(true_f)
    noisy_y_subopt = (np.abs(np.array(all_results_noisy_y[K_i]["f_mra_xk"]) - true_f)) / np.abs(true_f)
    high_y = max(eps_subopt[np.isfinite(eps_subopt)].max(), 
                 noisy_y_subopt[np.isfinite(noisy_y_subopt)].max())
    T = 0
    for i, K_i in enumerate(num_points):
        xs = np.array(subopt_of_best_feas_point_only(all_results_eps[K_i], true_f, b_norm))
        T = max(T, xs.size)
        ax_dict['D'].plot(xs, color=cmp[i], label=rf"$\epsilon_v={int(eps_sublevel * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.9, ls=lstyle[i])
        try:
            first_finite_idx = np.where(xs < np.inf)[0][0]
            if first_finite_idx >= 1:
                ax_dict['D'].vlines(first_finite_idx, ymin=xs[first_finite_idx], ymax=high_y, colors=cmp[i], alpha=0.9, lw=0.9, ls=lstyle[i])
                ax_dict['D'].text(first_finite_idx, high_y, r'$\infty$', color='k', fontsize=10, ha='center', va='bottom')
        except: pass 
        xs = np.array(subopt_of_best_feas_point_only(all_results_noisy_y[K_i], true_f, b_norm))
        T = max(T, xs.size)
        ax_dict['D'].plot(xs, color=cmp[2 + i], label=rf"$\epsilon_p={int(eps_sublevel * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.9, ls=lstyle[i])
        try:
            first_finite_idx = np.where(xs < np.inf)[0][0]
            if first_finite_idx >= 1:
                ax_dict['D'].vlines(first_finite_idx, ymin=xs[first_finite_idx], ymax=high_y, colors=cmp[2+i], alpha=0.9, lw=0.9, ls=lstyle[i])
                ax_dict['D'].text(first_finite_idx, high_y, r'$\infty$', color='k', fontsize=10, ha='center', va='bottom')
        except: pass
    ax_dict['D'].set_ylabel(r"$|f(x) + I_{-}(Ax - b) - f^\star|/|f^\star|$", fontsize=14)
    ax_dict['D'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['D'].set_yscale('log')
    ax_dict['D'].set_xlim(-(T-1)*0.05, (T-1) * 1.05)


    handles, labels = ax_dict['A'].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if filename is not None:
        plt.savefig(folder + "%s_all_metrics_4x.pdf" % filename, bbox_inches="tight")


def plot_eps_all_metrics_4x(all_results_eps, all_results_noisy_y, K_i, true_f, 
                                b_norm0, percents, filename=None, T=None, folder="../plots/"):
    cmp = ["red", "tomato", "blue", "royalblue"]
    lstyle = ["-", "--"]
    b_norm = 1 if b_norm0 == 0 else b_norm0
    mosaic = """
    AABB
    CCDD
    """
    fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=(12, 10), sharex=False, sharey=False)
    for i, eps in enumerate(percents):
        ax_dict['A'].plot((np.abs(np.array(all_results_eps[eps]["f_mra_xk"]) - true_f)) / np.abs(true_f),
                        color=cmp[i], label=rf"$\epsilon_v={int(eps * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
        ax_dict['A'].plot((np.abs(np.array(all_results_noisy_y[eps]["f_mra_xk"]) - true_f)) / np.abs(true_f),
                        color=cmp[2 + i], label=rf"$\epsilon_p={int(eps * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
    ax_dict['A'].set_ylabel(r"$|f(x) - f^\star|/|f^\star|$", fontsize=14)
    ax_dict['A'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['A'].set_yscale('log')

    for i, eps in enumerate(percents):
        ax_dict['B'].plot(np.array(all_results_eps[eps]["viol_primal_mra_xk"]) / b_norm,
                        color=cmp[i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
        ax_dict['B'].plot(np.array(all_results_noisy_y[eps]["viol_primal_mra_xk"]) / b_norm,
                        color=cmp[2 + i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
    if b_norm0 > 0:
        ax_dict['B'].set_ylabel(r'$r_p/\|b\|$', fontsize=14)
    else:
        ax_dict['B'].set_ylabel(r'$r_p$', fontsize=14)
    ax_dict['B'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['B'].set_yscale('log')

    for i, eps in enumerate(percents):
        ax_dict['C'].plot(np.array(all_results_eps[eps]["viol_primal_compl_mra_xk"]) / b_norm,
                        color=cmp[i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
        ax_dict['C'].plot(np.array(all_results_noisy_y[eps]["viol_primal_compl_mra_xk"]) / b_norm,
                        color=cmp[2 + i], alpha=0.9, lw=0.5, ls=lstyle[i], marker='.', markersize=2)
    if b_norm0 > 0:
        ax_dict['C'].set_ylabel(r'$(r_p+r_c)/\|b\|$', fontsize=14)
    else:
        ax_dict['C'].set_ylabel(r'$(r_p+r_c)$', fontsize=14)
    ax_dict['C'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['C'].set_yscale('log')
        
    eps_subopt = (np.abs(np.array(all_results_eps[eps]["f_mra_xk"]) - true_f)) / np.abs(true_f)
    noisy_y_subopt = (np.abs(np.array(all_results_noisy_y[eps]["f_mra_xk"]) - true_f)) / np.abs(true_f)
    high_y = max(eps_subopt[np.isfinite(eps_subopt)].max(), 
                    noisy_y_subopt[np.isfinite(noisy_y_subopt)].max())
    T = 0
    for i, eps in enumerate(percents):
        xs = np.array(subopt_of_best_feas_point_only(all_results_eps[eps], true_f, b_norm),)
        T = max(T, xs.size)
        ax_dict['D'].plot(xs, color=cmp[i], label=rf"$\epsilon_v={int(eps * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.9, ls=lstyle[i])
        try:
            first_finite_idx = np.where(xs < np.inf)[0][0]
            if first_finite_idx >= 1:
                ax_dict['D'].vlines(first_finite_idx, ymin=xs[first_finite_idx], ymax=high_y, colors=cmp[i], alpha=0.9, lw=0.9, ls=lstyle[i])
                ax_dict['D'].text(first_finite_idx, high_y, r'$\infty$', color='k', fontsize=10, ha='center', va='bottom')
        except: pass 
        xs = np.array(subopt_of_best_feas_point_only(all_results_noisy_y[eps], true_f, b_norm))
        T = max(T, xs.size)
        ax_dict['D'].plot(xs, color=cmp[2 + i], label=rf"$\epsilon_p={int(eps * 100)}\%, ~N={K_i}$",
                        alpha=0.9, lw=0.9, ls=lstyle[i])
        try:
            first_finite_idx = np.where(xs < np.inf)[0][0]
            if first_finite_idx >= 1:
                ax_dict['D'].vlines(first_finite_idx, ymin=xs[first_finite_idx], ymax=high_y, colors=cmp[2+i], alpha=0.9, lw=0.9, ls=lstyle[i])
                ax_dict['D'].text(first_finite_idx, high_y, r'$\infty$', color='k', fontsize=10, ha='center', va='bottom')
        except: pass
    ax_dict['D'].set_ylabel(r"$|f(x) + I_{-}(Ax - b) - f^\star|/|f^\star|$", fontsize=14)
    ax_dict['D'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['D'].set_yscale('log')
    ax_dict['D'].set_xlim(-(T-1)*0.05, (T-1) * 1.05)

    handles, labels = ax_dict['A'].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if filename is not None:
        plt.savefig(folder + "%s_all_metrics_4x.pdf" % filename, bbox_inches="tight")


def plot_all_methods_metrics_4x(all_results_eps, all_results_noisy_y, true_f, 
                                b_norm0, percent, eps_sublevel, filename=None, T=None, 
                                folder="../plots/", admm=False):
    b_norm = 1 if b_norm0 == 0 else b_norm0
    mosaic = """
    AABB
    CCDD
    """
    if len(all_results_noisy_y["subopt_xk"]) > len(all_results_eps["subopt_xk"]):
        all_res = all_results_noisy_y
    else:
        all_res = all_results_eps
    if T is None:
        T = len(all_res["viol_primal_mra_xk"])
    T = min(T, max(len(all_results_eps["f_mra_xk"]), len(all_results_noisy_y["f_mra_xk"])))
    cmp = ["orange", "red", "blue", "forestgreen", "violet"]
    fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=(12, 10), sharex=False, sharey=False)

    ax_dict['A'].plot((np.abs(np.array(all_res["f_xk"]) - true_f))[:T]/np.abs(true_f), 
                color=cmp[0], label=r"$x^k$", alpha=0.9, lw=0.5, marker='.', markersize=2) 
    ax_dict['A'].plot((np.abs(np.array(all_results_eps["f_mra_xk"])[:T] - true_f))/np.abs(true_f), 
                color=cmp[1], label=rf"$\bar x^k, ~\epsilon_v={int(eps_sublevel*100)}\%$", 
                alpha=0.9, lw=0.5, marker='.', markersize=2) 
    ax_dict['A'].plot((np.abs(np.array(all_results_noisy_y["f_mra_xk"])[:T] - true_f))/np.abs(true_f), 
                color=cmp[2], label=rf"$\bar x^k, ~\epsilon_p={int(percent*100)}\%$", 
                alpha=0.9, lw=0.5, marker='.', markersize=2) 
    ax_dict['A'].plot((np.abs(np.array(all_res["f_paver_xk"])[:T] - true_f))/np.abs(true_f), 
                color=cmp[3], label=r"$\frac{1}{k}\sum_{j=1}^k x^j$", 
                alpha=0.9, lw=0.5, marker='.', markersize=2) 
    ax_dict['A'].plot((np.abs(np.array(all_res["f_proj_xk"])[:T] - true_f))/np.abs(true_f), 
                color=cmp[4], label=r"$\Pi(x^k)$", 
                alpha=0.9, lw=0.5, marker='.', markersize=2) 
    ax_dict['A'].set_ylabel(r"$|f(x) - f^\star|/|f^\star|$", fontsize=14)
    ax_dict['A'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['A'].set_yscale('log')

    ax_dict['B'].plot(np.array(all_res["viol_primal_xk"])[:T]/ b_norm, 
                    color=cmp[0], label=r"$x^k$", alpha=0.9, lw=0.5, marker='.', markersize=2)  
    ax_dict['B'].plot(np.array(all_results_eps["viol_primal_mra_xk"])[:T] / b_norm, 
                color=cmp[1], label=rf"$\bar x^k, ~\epsilon_v={int(eps_sublevel*100)}\%$", alpha=0.9, lw=0.5, marker='.', markersize=2) 
    ax_dict['B'].plot(np.array(all_results_noisy_y["viol_primal_mra_xk"])[:T] / b_norm,
                color=cmp[2], label=rf"$\bar x^k, ~\epsilon_p={int(percent*100)}\%$", alpha=0.9, lw=0.5, marker='.', markersize=2) 
    ax_dict['B'].plot(np.array(all_res["viol_primal_paver_xk"])[:T] / b_norm, 
                color=cmp[3], label=r"$\frac{1}{k}\sum_{j=1}^k x^j$", alpha=0.9, lw=0.5, marker='.', markersize=2) 

    if b_norm0 > 0 and not admm:
        ax_dict['B'].set_ylabel(r'$r_p/\|b\|$', fontsize=14)
    else:
        ax_dict['B'].set_ylabel(r'$r_p$', fontsize=14)
    ax_dict['B'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['B'].set_yscale('log')


    ax_dict['C'].plot(np.array(all_res["viol_primal_compl_xk"])[:T] / b_norm,  
                    color=cmp[0], label=r"$x^k$", alpha=0.9, lw=0.5, marker='.', markersize=2)  
    ax_dict['C'].plot(np.array(all_results_eps["viol_primal_compl_mra_xk"])[:T] / b_norm, 
                    color=cmp[1], label=rf"$\bar x^k, ~\epsilon_v={int(eps_sublevel*100)}\%$", 
                    alpha=0.9, lw=0.5, marker='.', markersize=2) 
    ax_dict['C'].plot(np.array(all_results_noisy_y["viol_primal_compl_mra_xk"])[:T] / b_norm,
                    color=cmp[2], label=rf"$\bar x^k, ~\epsilon_p={int(percent*100)}\%$", 
                    alpha=0.9, lw=0.5, marker='.', markersize=2)
    if admm:
        ax_dict['C'].set_ylabel(r'$r_p+r_d$', fontsize=14)
    else:
        if b_norm0 > 0:
            ax_dict['C'].set_ylabel(r'$(r_p+r_c)/\|b\|$', fontsize=14)
        else:
            ax_dict['C'].set_ylabel(r'$(r_p+r_c)$', fontsize=14)
    ax_dict['C'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['C'].set_yscale('log')

    eps_subopt = (np.abs(np.array(all_results_eps["f_mra_xk"]) - true_f)) / np.abs(true_f)
    noisy_y_subopt = (np.abs(np.array(all_results_noisy_y["f_mra_xk"]) - true_f)) / np.abs(true_f)
    xk_subopt = (np.abs(np.array(all_res["f_xk"]) - true_f)) / np.abs(true_f)
    paverage_subopt = (np.abs(np.array(all_res["f_paver_xk"]) - true_f)) / np.abs(true_f)
    proj_subopt = (np.abs(np.array(all_res["f_proj_xk"]) - true_f)) / np.abs(true_f)

    high_y = max(eps_subopt[np.isfinite(eps_subopt)].max(), 
                noisy_y_subopt[np.isfinite(noisy_y_subopt)].max(),
                xk_subopt[np.isfinite(xk_subopt)].max(),
                paverage_subopt[np.isfinite(paverage_subopt)].max(),
                )
    proj_high_y = proj_subopt[np.isfinite(proj_subopt)].max()
    if proj_high_y < np.inf:
        high_y = max(high_y, proj_high_y)

    labels = ["$x^k$", 
            rf"$\bar x^k, ~\epsilon_v={int(eps_sublevel*100)}\%$", 
            rf"$\bar x^k, ~\epsilon_p={int(percent*100)}\%$",
            r"$\frac{1}{k}\sum_{j=1}^k x^j$",
            r"$\Pi(x^k)$"]
    labels_fs = {"$x^k$":all_res["f_xk"][:T], 
            rf"$\bar x^k, ~\epsilon_v={int(eps_sublevel*100)}\%$":all_results_eps["f_mra_xk"][:T], 
            rf"$\bar x^k, ~\epsilon_p={int(percent*100)}\%$":all_results_noisy_y["f_mra_xk"][:T],
            r"$\frac{1}{k}\sum_{j=1}^k x^j$":all_res["f_paver_xk"][:T],
            r"$\Pi(x^k)$":all_res["f_proj_xk"][:T]}
    labels_rs = {"$x^k$":all_res["viol_primal_xk"][:T], 
            rf"$\bar x^k, ~\epsilon_v={int(eps_sublevel*100)}\%$":all_results_eps["viol_primal_mra_xk"][:T], 
            rf"$\bar x^k, ~\epsilon_p={int(percent*100)}\%$":all_results_noisy_y["viol_primal_mra_xk"][:T],
            r"$\frac{1}{k}\sum_{j=1}^k x^j$":all_res["viol_primal_paver_xk"][:T],
            r"$\Pi(x^k)$":np.zeros(T)}

    for i, label in enumerate(labels):
        xs = np.array(subopt_of_best_feas_point_base(labels_fs[label], labels_rs[label], true_f, b_norm))
        ax_dict['D'].plot(xs, color=cmp[i], label=label, alpha=0.9, lw=0.9)
        try:
            first_finite_idx = np.where(xs < np.inf)[0][0]
            if first_finite_idx >= 1:
                ax_dict['D'].vlines(first_finite_idx, ymin=xs[first_finite_idx], ymax=high_y, 
                                colors=cmp[i], alpha=0.9, lw=0.9)
                ax_dict['D'].text(first_finite_idx, high_y, r'$\infty$', color='k', 
                                  fontsize=10, ha='center', va='bottom')
        except: pass
    if admm:
        ax_dict['D'].set_ylabel(r"$|f(x) + I_{=0}(x - Ez) - f^\star|/|f^\star|$", fontsize=14)
    else:
        ax_dict['D'].set_ylabel(r"$|f(x) + I_{-}(Ax - b) - f^\star|/|f^\star|$", fontsize=14)
    ax_dict['D'].set_xlabel(r'$k$', fontsize=14)
    ax_dict['D'].set_yscale('log')
    ax_dict['D'].set_xlim(-(T-1)*0.05, (T-1) * 1.05)

    handles, labels = ax_dict['A'].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1), fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if filename is not None:
        plt.savefig(folder + "%s_subopt_res_square.pdf" % filename, bbox_inches="tight")
