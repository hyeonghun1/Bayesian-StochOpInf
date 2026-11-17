"""
Derivative-based vs GP-BayesOpInf vs trajectory-based likelihood for Bayesian OpInf, on the toy ROMs.

All three methods see the same noisy snapshots (n_traj trajectories x K snapshots on [0, T_train]):

  derivative  Guo, McQuarrie & Willcox (closed form, row-wise): D from the RAW noisy snapshots,
              local-polynomial derivative estimates, Tikhonov prior with per-row lam_i.
              * with mcmc_derivative_derv_fixed.py in the folder (--data base, the default when it
                imports): exactly that script's build_problem (De Brabanter derivatives, Guo's lam fixed
                point), i.e. the same data and baseline as mcmc_trajectory2.py
              * otherwise (--data local): Savitzky-Golay derivatives (window by GCV, cubic) and
                per-row lam_i, sigma_i^2 by maximum marginal likelihood (evidence fixed point)
  gp          GP-BayesOpInf (McQuarrie, Chaudhuri, Willcox & Guo), the authors' code unchanged, via
              gp_bayesopinf_toy.py: GP-smoothed states in D~, GP derivatives, GLS weights from the GP
              derivative covariance, regularizer by their stability-constrained search
  gp_tuned    (optional, --methods ... gp_tuned) the same code, but the floor eta in the weights
              W = (C + eta I)^{-1/2} is tuned instead of fixed at their 1e-8: candidates 1e-8 and
              --gp_eta_grid x max eig(C), chosen jointly with the regularizer by THEIR training-data
              criterion (stable on the prediction domain, lowest error against the GP states on the
              training window; posterior-mean operator), then their regularizer search runs with it.
              Not part of their paper; it tests whether the GP method's failure is just the eta floor
  --gp_fallback  (off by default; NOT part of their method) when their regularizer search raises
              "Regularization search optimization FAILED" (no regularizer whose 20 posterior draws all stay
              bounded on the prediction domain), use the regularizer their grid search picked instead of
              stopping the run. Reported in the output and the json; files get the suffix _gpfallback
  trajectory  run_trajectory_based_likelihood_method from mcmc_trajectory2.py (single shooting,
              Laplace warm start, NUTS), with that script's default prior (the derivative method's
              Tikhonov prior)

Scoring
  operators   relative error of the posterior mean, 95% coverage of the true entries, mean 95% width
  prediction  n_pf posterior draws rolled out (RK4, 4 substeps per snapshot interval) on [0, T_test]
              from the CLEAN initial condition, so the bands show operator uncertainty only; for the
              n_traj training ICs and n_test unseen ICs; median relative error, 95% band coverage of
              the clean truth and band width, separately on the training window [0, T_train] and the
              extrapolation window (T_train, T_test]; a draw that leaves 5x the truth's range or
              blows up counts as unstable and is excluded from the bands (stable fraction reported)
  GP diagnostics  error of the GP state / derivative estimates, and the operator error of ordinary
              least squares with (GP states, true derivatives), (true states, GP derivatives) and
              (GP states, GP derivatives): splits the GP method's error into the part that comes
              from D~ (errors in variables after smoothing), from the GP derivatives, and from the
              GLS weighting

Usage (put next to toy_systems.py, mcmc_trajectory2.py, gp_bayesopinf_toy.py and the GP-BayesOpInf
clone; mcmc_derivative_derv_fixed.py optional):
    git clone https://github.com/sandialabs/GP-BayesOpInf.git
    pip install opinf==0.5.9 scikit-learn h5py
    python compare_gp_trajectory.py --system damped --noise 0.05
    python compare_gp_trajectory.py --system limit_cycle --noise 0.10 --data local
    python compare_gp_trajectory.py --system damped --noise 0.05 --methods gp_tuned   # adds to the same table
Results: pdfs/compare_gp_trajectory_pdfs/<tag>.json plus figures; summarize with
    python compare_gp_trajectory.py --summarize
"""

import argparse
import glob
import json
import os
import sys
import time
import types

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_coeffs, savgol_filter

import toy_systems as ts

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "pdfs", "compare_gp_trajectory_pdfs")
r = ts.r
ALL_METHODS = ("derivative", "gp", "gp_tuned", "trajectory")      # table / figure order
DEFAULT_METHODS = ("derivative", "gp", "trajectory")
LABELS = dict(derivative="derivative (Guo)", gp="GP-BayesOpInf", gp_tuned="GP-BayesOpInf, eta tuned",
              trajectory="trajectory (NUTS)")
COLORS = dict(derivative="#8a8a8a", gp="#d4762c", gp_tuned="#8e5bb5", trajectory="#2a6fb5")


# ---------------------------------------------------------------------------------------------
# Data and the derivative baseline
# ---------------------------------------------------------------------------------------------

def _import_base():
    try:
        import mcmc_derivative_derv_fixed as base
        return base
    except ImportError:
        return None


def _import_trajectory_module(base):
    """mcmc_trajectory2 imports mcmc_derivative_derv_fixed at module level but the trajectory sampler
    only needs r from it; without that script, register a stand-in that provides r."""
    if base is None and "mcmc_derivative_derv_fixed" not in sys.modules:
        stub = types.ModuleType("mcmc_derivative_derv_fixed")
        stub.r = r
        stub.SYSTEMS = ts.SYSTEMS
        stub.system_tag = ts.system_tag
        stub.rom_rhs = ts.rom_rhs
        sys.modules["mcmc_derivative_derv_fixed"] = stub
    import mcmc_trajectory2
    return mcmc_trajectory2


def local_poly_derivatives(Q, dt, poly=3, candidates=range(7, 61, 2)):
    """Savitzky-Golay derivative estimates of the rows of Q (r x K); window by GCV on the states."""
    K = Q.shape[1]
    best, window = np.inf, None
    for w in candidates:
        if w <= poly or w > K:
            continue
        h0 = savgol_coeffs(w, poly)[w // 2]
        rss = np.mean((Q - savgol_filter(Q, w, poly, axis=1, mode="interp"))**2)
        if rss / (1.0 - h0)**2 < best:
            best, window = rss / (1.0 - h0)**2, w
    return savgol_filter(Q, window, poly, deriv=1, delta=dt, axis=1, mode="interp"), window


def evidence_posterior(D, y, iters=500, tol=1e-10):
    """Bayesian linear regression y ~ N(D o, 1/beta), o ~ N(0, 1/alpha I), with alpha, beta by maximum
    marginal likelihood (MacKay fixed point). In Guo et al.'s notation lam^2 = alpha / beta,
    sigma^2 = 1 / beta. Returns mu, Sigma, lam, sigma2."""
    N, d = D.shape
    G = D.T @ D
    e = np.linalg.eigvalsh(G)
    Dy = D.T @ y
    alpha, beta = 1e-2, 1.0 / np.var(y)
    for _ in range(iters):
        A = alpha * np.eye(d) + beta * G
        m = beta * np.linalg.solve(A, Dy)
        gamma = np.sum(beta * e / (alpha + beta * e))
        alpha_new = gamma / (m @ m)
        beta_new = (N - gamma) / np.sum((y - D @ m)**2)
        done = abs(np.log(alpha_new / alpha)) < tol and abs(np.log(beta_new / beta)) < tol
        alpha, beta = alpha_new, beta_new
        if done:
            break
    A = alpha * np.eye(d) + beta * G
    Sigma = np.linalg.inv(A)
    return beta * Sigma @ Dy, Sigma, np.sqrt(alpha / beta), 1.0 / beta


def data_matrix(Q):
    """[1, q, q_i q_j (i <= j)] for Q (r x K), toy_systems order (K x d)."""
    iu, ju = np.triu_indices(r)
    return np.hstack([np.ones((Q.shape[1], 1)), Q.T, (Q[iu] * Q[ju]).T])


def build_local(system, noise, T_train, T_test, K, n_traj, seed):
    """Toy data as in mcmc_trajectory2 (relative noise = noise x pooled clean training range of each
    variable, ICs ~ U(-1.5, 1.5)^r) and the local derivative baseline."""
    O_true = ts.operators_to_matrix(*ts.true_operators(system))
    rng = np.random.default_rng(seed)
    t = np.linspace(0, T_train, K)
    dt = t[1] - t[0]
    n_ext = int(round((T_test - T_train) / dt))
    t_full = np.concatenate([t, T_train + dt * np.arange(1, n_ext + 1)])
    clean = np.stack([solve_ivp(ts.rom_rhs(O_true), (0, t_full[-1]), rng.uniform(-1.5, 1.5, r),
                                t_eval=t_full, rtol=1e-10, atol=1e-12).y for _ in range(n_traj)])
    noise_std = noise * np.ptp(clean[:, :, :K].transpose(1, 0, 2).reshape(r, -1), axis=1)
    Qs = clean[:, :, :K] + noise_std[None, :, None] * rng.standard_normal((n_traj, r, K))

    dQ, windows = zip(*[local_poly_derivatives(Q, dt) for Q in Qs])
    D = np.vstack([data_matrix(Q) for Q in Qs])                 # raw noisy states, as Guo et al.
    R = np.hstack(dQ)
    post = [evidence_posterior(D, R[i]) for i in range(r)]
    return dict(t=t, t_full=t_full, Qs=Qs, Qs_clean_full=clean, O_true=O_true, noise_std=noise_std,
                mu=np.array([p[0] for p in post]), Sigma=np.array([p[1] for p in post]),
                lam=np.array([p[2] for p in post]), sigma2=np.array([p[3] for p in post]),
                deriv_desc=f"Savitzky-Golay (cubic, GCV windows {sorted(set(windows))} pts), "
                           "lam by max. marginal likelihood")


def build_base(base, system, noise, T_train, T_test, K, n_traj):
    """Same data and derivative baseline as mcmc_trajectory2.py with its default options."""
    prob = base.build_problem(noise=noise, T_train=T_train, K=K, n_traj=n_traj, system=system,
                              deriv="dbr", k_dbr="auto", dbr_edge="trim", lam=None, prior="tikhonov",
                              noise_ref="pooled", clean_ic=False, uniform_bound=10.0, T_test=T_test,
                              hs_p0=3.0)
    prob = dict(prob)
    prob["Qs_clean_full"] = np.asarray(prob["Qs_clean_full"])
    prob["deriv_desc"] = "mcmc_derivative_derv_fixed.build_problem (De Brabanter, Guo lam fixed point)"
    return prob


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------

def draws_from_rows(mu, Sigma, n, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.multivariate_normal(mu[i], Sigma[i], n) for i in range(r)], axis=1)


def score_operators(O_s, O_true):
    m = O_s.mean(0)
    lo, hi = np.percentile(O_s, [2.5, 97.5], axis=0)
    return dict(op_err=float(np.linalg.norm(m - O_true) / np.linalg.norm(O_true)),
                op_cov=float(np.mean((O_true >= lo) & (O_true <= hi))),
                op_width=float(np.mean(hi - lo)))


def rollout(O_s, q0, t, substeps=4):
    """Fixed-step RK4 for a batch of operators O_s (n, r, d) from q0 (r,); returns (n, r, len(t)),
    NaN after a blow-up."""
    iu, ju = np.triu_indices(r)

    def f(q):
        dq = np.concatenate([np.ones((q.shape[0], 1)), q, q[:, iu] * q[:, ju]], axis=1)
        return np.einsum("nrd,nd->nr", O_s, dq)

    n = len(O_s)
    q = np.repeat(q0[None, :], n, axis=0).astype(float)
    out = np.empty((n, r, len(t)))
    out[:, :, 0] = q
    with np.errstate(all="ignore"):
        for k in range(1, len(t)):
            h = (t[k] - t[k - 1]) / substeps
            for _ in range(substeps):
                k1 = f(q); k2 = f(q + 0.5 * h * k1); k3 = f(q + 0.5 * h * k2); k4 = f(q + h * k3)
                q = q + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                q[~np.all(np.isfinite(q), axis=1)] = np.nan
                q[np.abs(q).max(axis=1) > 1e6] = np.nan
            out[:, :, k] = q
    return out


def score_prediction(Y, truth, t, T_train):
    """Y (n, r, k) rollouts from the clean IC, truth (r, k)."""
    span = np.ptp(truth, axis=1).max()
    centre = truth.mean(axis=1, keepdims=True)
    stable = np.all(np.isfinite(Y), axis=(1, 2)) & (np.abs(Y - centre).max(axis=(1, 2)) <= 5 * span)
    out = dict(stable=float(stable.mean()))
    Ys = Y[stable]
    for name, m in [("train", t <= T_train + 1e-12), ("extrap", t > T_train + 1e-12)]:
        if len(Ys) < 10:
            out.update({f"{name}_err": np.nan, f"{name}_cov": np.nan, f"{name}_width": np.nan})
            continue
        lo, med, hi = np.percentile(Ys[:, :, m], [2.5, 50, 97.5], axis=0)
        tr = truth[:, m]
        out[f"{name}_err"] = float(np.linalg.norm(med - tr) / np.linalg.norm(tr))
        out[f"{name}_cov"] = float(np.mean((tr >= lo) & (tr <= hi)))
        out[f"{name}_width"] = float(np.mean(hi - lo))
    return out


def average_scores(rows):
    keys = rows[0].keys()
    return {k: float(np.nanmean([s[k] for s in rows])) for k in keys}


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------

def plot_operators(draws, O_true, info, path):
    import matplotlib.pyplot as plt
    d = O_true.shape[1]
    iu, ju = np.triu_indices(r)
    names = ["c"] + [f"A{k}" for k in range(r)] + [f"H{i}{j}" for i, j in zip(iu, ju)]
    methods = [m for m in ALL_METHODS if m in draws]
    fig, axes = plt.subplots(r, 1, figsize=(11, 2.6 * r), sharex=True)
    for i, ax in enumerate(axes):
        for k, meth in enumerate(methods):
            x = np.arange(d) + (k - (len(methods) - 1) / 2) * 0.2
            m = draws[meth][:, i].mean(0)
            lo, hi = np.percentile(draws[meth][:, i], [2.5, 97.5], axis=0)
            ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o", ms=3, lw=1.4, color=COLORS[meth],
                        label=LABELS[meth] if i == 0 else None)
        ax.plot(np.arange(d), O_true[i], "kx", ms=8, mew=2, label="truth" if i == 0 else None)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_ylabel(f"row {i}")
    axes[-1].set_xticks(np.arange(d), names)
    axes[0].legend(ncol=4, fontsize=15, loc="upper right")
    fig.suptitle(info + "  |  posterior means and 95% intervals", fontsize=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_predictions(cases, t, T_train, info, path):
    """cases: list of (title, truth (r, k), noisy obs (r, K) or None, {method: rollouts})."""
    import matplotlib.pyplot as plt
    methods = [m for m in ALL_METHODS if m in cases[0][3]]
    fig, axes = plt.subplots(r, len(cases) * len(methods), figsize=(3.0 * len(cases) * len(methods), 2.1 * r + 0.8),
                             sharex=True, squeeze=False)
    for c, (title, truth, obs, Ys) in enumerate(cases):
        for k, meth in enumerate(methods):
            col = c * len(methods) + k
            Y = Ys[meth]
            span = np.ptp(truth, axis=1).max()
            ok = np.all(np.isfinite(Y), axis=(1, 2)) & (np.abs(Y - truth.mean(1, keepdims=True)).max(axis=(1, 2)) <= 5 * span)
            for j in range(r):
                ax = axes[j, col]
                if ok.sum() >= 10:
                    lo, med, hi = np.percentile(Y[ok], [2.5, 50, 97.5], axis=0)
                    ax.fill_between(t, lo[j], hi[j], color=COLORS[meth], alpha=0.35, lw=0)
                    ax.plot(t, med[j], color=COLORS[meth], lw=1)
                if obs is not None:
                    ax.plot(t[:obs.shape[1]], obs[j], ".", color="k", ms=1.2, alpha=0.35)
                ax.plot(t, truth[j], "k-", lw=0.9)
                ax.axvline(T_train, color="0.5", lw=0.8, ls=":")
                pad = 0.2 * np.ptp(truth[j]) + 1e-3
                ax.set_ylim(truth[j].min() - pad, truth[j].max() + pad)
                if col == 0:
                    ax.set_ylabel(f"$q_{j}$")
            axes[0, col].set_title(f"{title}\n{LABELS[meth]} (stable {ok.mean():.0%})", fontsize=12)
            axes[-1, col].set_xlabel("t")
    fig.suptitle(info + "  |  95% bands from the clean IC (black: truth, dots: training data)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)



####### Main #######
def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    base = _import_base() if args.data in ("auto", "base") else None
    if args.data == "base" and base is None:
        raise ImportError("--data base needs mcmc_derivative_derv_fixed.py in this folder")
    mt2 = _import_trajectory_module(base)
    from gp_bayesopinf_toy import gp_bayesopinf

    methods = [m for m in ALL_METHODS if m in args.methods]
    base_tag = f"{ts.system_tag(args.system)}_noise{args.noise:g}_ntraj{args.n_traj}_K{args.K}_Ttrain{args.T_train:g}_seed{args.seed}"
    if args.gp_fallback and any(m in methods for m in ("gp", "gp_tuned")):
        base_tag += "_gpfallback"           # never overwrite runs of their unmodified search
    tag = base_tag + ("" if tuple(methods) == DEFAULT_METHODS else "_" + "-".join(methods)) \

    t_start = time.time()
    if base is not None:
        prob = build_base(base, args.system, args.noise, args.T_train, args.T_test, args.K, args.n_traj)
        data_mode = "base"
    else:
        prob = build_local(args.system, args.noise, args.T_train, args.T_test, args.K, args.n_traj, args.seed)
        data_mode = "local"
    O_true, t, Qs = prob["O_true"], prob["t"], np.asarray(prob["Qs"])
    t_full = prob["t_full"]
    dt = t[1] - t[0]
    n_traj, K = Qs.shape[0], Qs.shape[2]
    print(f"\n=== {tag} ===\ndata: {data_mode}; derivative baseline: {prob['deriv_desc']}\n"
          f"{n_traj} trajectories x {K} snapshots on [0, {args.T_train:g}], noise std "
          f"{np.array2string(np.asarray(prob['noise_std']), precision=4)}, predict to {args.T_test:g}")

    draws, info, timing = {}, {}, {}

    # derivative-based: exact Gaussian posterior
    if "derivative" in methods:
        draws["derivative"] = draws_from_rows(prob["mu"], prob["Sigma"], args.n_draws, args.seed)
        timing["derivative"] = 0.0

    # GP-BayesOpInf (authors' code): their eta = 1e-8 ("gp") and/or eta tuned by their training-error
    # criterion ("gp_tuned"); each variant fits its own GPs
    gp = None
    for meth in [m for m in ("gp", "gp_tuned") if m in methods]:
        grid = args.gp_eta_grid if meth == "gp_tuned" else None
        t0 = time.time()
        res = gp_bayesopinf(t, Qs, t_full, repo_dir=args.gp_repo, n_restarts=args.gp_restarts, eta_grid=grid,
                            fallback=args.gp_fallback,
                            log_path=os.path.join(OUT_DIR, f"gplog_{base_tag}" + ("_tuned" if grid else "") + ".txt"))
        timing[meth] = time.time() - t0
        draws[meth] = draws_from_rows(res["mu"], res["Sigma"], args.n_draws, args.seed + 1)
        info[f"{meth}_eta"], info[f"{meth}_reg"], info[f"{meth}_eta_screen"] = res["eta"], res["reg"], res["eta_screen"]
        info[f"{meth}_fallback"] = res["fallback"]
        if res["fallback"] is not None:
            print(f"{LABELS[meth]}: their regularizer search FAILED (no stable regularizer in the refinement); "
                  f"--gp_fallback used the grid-search pick {res['fallback']:.3e} (not their method)")
        if res["eta_screen"]:
            print("eta screening (eta / max eig C: best reg, training error of the mean ROM): "
                  + ", ".join(f"{rel:.0e}: {reg:.1e}, {err:.3f}" for e, rel, reg, err in res["eta_screen"]))
        print(f"{LABELS[meth]}: {timing[meth]:.0f}s, eta {res['eta']:.2e}, regularizer {res['reg']:.3e}, "
              f"{len(res['refits'])} GP refits ({sum(rec[3] > rec[2] for rec in res['refits'])} improved, "
              f"{sum(rec[4] for rec in res['refits'])} still flagged), sqrt(chi) / true noise std (median over "
              f"trajectories) = {np.array2string(np.median(np.sqrt(res['gp_params'][:, :, 2]), axis=0) / np.asarray(prob['noise_std']), precision=3)}")
        gp = gp or res

    # GP diagnostics: where does the GP method's operator error come from? (GP estimates do not
    # depend on eta; only the weights do)
    if gp is not None:
        gp_diagnostics(gp, prob, K, info)

    run_rest(args, methods, base_tag, tag, prob, draws, info, timing, mt2, t_start, data_mode)


def gp_diagnostics(gp, prob, K, info):
    O_true = prob["O_true"]
    clean = prob["Qs_clean_full"][:, :, :K]
    f_true = ts.rom_rhs(O_true)
    dclean = np.stack([np.array([f_true(0, q) for q in Q.T]).T for Q in clean])
    ols = lambda D, z: np.linalg.lstsq(D, z.T, rcond=None)[0].T
    D_true = np.vstack([data_matrix(Q) for Q in clean])
    D_gp = np.vstack([data_matrix(Q) for Q in gp["states"]])
    z_true, z_gp = np.hstack(list(dclean)), np.hstack(list(gp["ddts"]))
    rel = lambda A, B: float(np.linalg.norm(A - B) / np.linalg.norm(B))
    info["gp_diag"] = dict(
        state_rmse_over_noise=(np.sqrt(np.mean((gp["states"] - clean)**2, axis=(0, 2))) /
                               np.asarray(prob["noise_std"])).tolist(),
        deriv_rel_err=rel(gp["ddts"], dclean),
        ols_gp_states_true_derivs=rel(ols(D_gp, z_true), O_true),
        ols_true_states_gp_derivs=rel(ols(D_true, z_gp), O_true),
        ols_gp_states_gp_derivs=rel(ols(D_gp, z_gp), O_true),
        gls_this_run=rel(gp["mu"], O_true),
        reg=gp["reg"], eta=gp["eta"], lml=gp["gp_lml"].tolist(), gp_params=gp["gp_params"].tolist(),
        refits=gp["refits"])
    print("GP diagnostics: operator error of OLS with (GP states, true derivs) "
          f"{info['gp_diag']['ols_gp_states_true_derivs']:.3f} | (true states, GP derivs) "
          f"{info['gp_diag']['ols_true_states_gp_derivs']:.3f} | (GP, GP) "
          f"{info['gp_diag']['ols_gp_states_gp_derivs']:.3f} | GLS (eta {gp['eta']:.1e}) "
          f"{info['gp_diag']['gls_this_run']:.3f}")


def run_rest(args, methods, base_tag, tag, prob, draws, info, timing, mt2, t_start, data_mode):
    O_true, t, Qs, t_full = prob["O_true"], prob["t"], np.asarray(prob["Qs"]), prob["t_full"]
    dt = t[1] - t[0]
    n_traj, K = Qs.shape[0], Qs.shape[2]

    # trajectory-based: NUTS with mcmc_trajectory2's default (Tikhonov) prior
    if "trajectory" in methods:
        sd_row = np.sqrt(np.asarray(prob["sigma2"])) / np.asarray(prob["lam"])
        tp_sd = np.repeat(sd_row[:, None], O_true.shape[1], axis=1)
        t0 = time.time()
        O_traj, sig_y, _, diag = mt2.run_trajectory_based_likelihood_method(
            Qs.transpose(0, 2, 1), dt, np.asarray(prob["mu"]), prior="normal", prior_sd=tp_sd,
            substeps=args.substeps, seed=args.seed)
        timing["trajectory"] = time.time() - t0
        draws["trajectory"] = O_traj
        info["nuts"] = dict(div=diag["div"], ess_min=diag["ess_min"], rhat_max=diag["rhat_max"],
                            steps=diag["steps"], sigma_y=sig_y.mean(0).tolist())
        print(f"trajectory NUTS: {timing['trajectory']:.0f}s, {diag['div']} divergences, min ESS {diag['ess_min']:.0f}, "
              f"max R-hat {diag['rhat_max']:.3f}, sigma_y {np.array2string(sig_y.mean(0), precision=4)}")

    # unseen test ICs (separate stream: identical for every method and data mode)
    rng_test = np.random.default_rng(1000 + args.seed)
    test_truth = np.stack([solve_ivp(ts.rom_rhs(O_true), (0, t_full[-1]), rng_test.uniform(-1.5, 1.5, r),
                                     t_eval=t_full, rtol=1e-10, atol=1e-12).y for _ in range(args.n_test)])

    rng_pf = np.random.default_rng(7 + args.seed)
    results, rollouts = {}, {}
    for meth in methods:
        O_s = draws[meth]
        O_pf = O_s[rng_pf.choice(len(O_s), min(args.n_pf, len(O_s)), replace=False)]
        res = dict(score_operators(O_s, O_true), time=timing[meth])
        for group, truths in [("trainIC", prob["Qs_clean_full"]), ("testIC", test_truth)]:
            sc = []
            for l, tr in enumerate(truths):
                Y = rollout(O_pf, tr[:, 0], t_full)
                sc.append(score_prediction(Y, tr, t_full, args.T_train))
                if l < args.n_show:
                    rollouts[(meth, group, l)] = Y
            res[group] = average_scores(sc)
        results[meth] = res

    hdr = (f"\n{'method':>18s} | {'op err':>6s} {'op cov':>6s} {'op wid':>6s} | "
           f"{'IC':>7s} {'stable':>6s} | {'err tr':>6s} {'cov tr':>6s} | {'err ex':>6s} {'cov ex':>6s} {'wid ex':>7s}")
    print(hdr); print("-" * len(hdr))
    for meth in methods:
        s = results[meth]
        for g in ("trainIC", "testIC"):
            p = s[g]
            head = (f"{LABELS[meth]:>18s} | {s['op_err']:6.3f} {s['op_cov']:6.2f} {s['op_width']:6.3f} | "
                    if g == "trainIC" else f"{'':>18s} | {'':>20s} | ")
            print(head + f"{g[:-2]:>7s} {p['stable']:6.2f} | {p['train_err']:6.3f} {p['train_cov']:6.2f} | "
                         f"{p['extrap_err']:6.3f} {p['extrap_cov']:6.2f} {p['extrap_width']:7.4f}")

    info_str = (f"{args.system}, noise {100 * args.noise:g}%, {n_traj} traj. x {K} snapshots, "
                f"T_train {args.T_train:g}")
    plot_operators(draws, O_true, info_str, os.path.join(OUT_DIR, f"operators_{tag}.pdf"))
    # one file per IC index l: training IC l next to unseen IC l (whichever exist)
    pred_files = []
    for l in range(max(1, min(args.n_show, max(n_traj, args.n_test)))):
        cases = []
        if l < n_traj:
            cases.append((f"training IC {l}", prob["Qs_clean_full"][l], Qs[l],
                          {m: rollouts[(m, "trainIC", l)] for m in methods}))
        if l < args.n_test:
            cases.append((f"unseen IC {l}", test_truth[l], None, {m: rollouts[(m, "testIC", l)] for m in methods}))
        fname = f"predictions_{tag}.pdf" if args.n_show <= 1 else f"predictions_{tag}_ic{l}.pdf"
        plot_predictions(cases, t_full, args.T_train, info_str, os.path.join(OUT_DIR, fname))
        pred_files.append(fname)

    out = dict(tag=tag, base_tag=base_tag, args=vars(args), data_mode=data_mode, deriv_desc=prob["deriv_desc"],
               noise_std=np.asarray(prob["noise_std"]).tolist(), results=results, info=info,
               op_means={m: draws[m].mean(0).tolist() for m in methods}, O_true=O_true.tolist(),
               total_time=time.time() - t_start)
    with open(os.path.join(OUT_DIR, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {tag}.json, operators_{tag}.pdf, {', '.join(pred_files)} in {OUT_DIR} "
          f"({out['total_time'] / 60:.1f} min)")


def summarize():
    """Markdown table over every saved run."""
    merged = {}                       # runs with the same data (base tag) are merged, e.g. a gp_tuned rerun
    for p in sorted(glob.glob(os.path.join(OUT_DIR, "*.json"))):
        with open(p) as f:
            o = json.load(f)
        key = o.get("base_tag", o["tag"])
        if key not in merged:
            merged[key] = dict(args=o["args"], results={})
        merged[key]["results"].update(o["results"])
    rows = sorted(merged.values(), key=lambda o: (o["args"]["system"], o["args"]["noise"], o["args"]["seed"]))
    lines = ["| system | noise | seed | method | op err | op cov | op width | stable (train/test IC) | "
             "extrap err train IC | extrap cov train IC | extrap err test IC | extrap cov test IC | time [s] |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in rows:
        a = o["args"]
        for m in [m for m in ALL_METHODS if m in o["results"]]:
            s = o["results"][m]
            tr, te = s["trainIC"], s["testIC"]
            lines.append(f"| {a['system']} | {100 * a['noise']:g}% | {a['seed']} | {LABELS[m]} | {s['op_err']:.3f} | "
                         f"{s['op_cov']:.2f} | {s['op_width']:.3f} | {tr['stable']:.2f} / {te['stable']:.2f} | "
                         f"{tr['extrap_err']:.3f} | {tr['extrap_cov']:.2f} | {te['extrap_err']:.3f} | "
                         f"{te['extrap_cov']:.2f} | {s['time']:.0f} |")
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="damped", choices=list(ts.SYSTEMS))
    ap.add_argument("--noise", type=float, default=0.05, help="relative noise (std = noise x clean range)")
    ap.add_argument("--T_train", type=float, default=6.0)
    ap.add_argument("--T_test", type=float, default=12.0)
    ap.add_argument("--K", type=int, default=300, help="snapshots per trajectory")
    ap.add_argument("--n_traj", type=int, default=10)
    ap.add_argument("--n_test", type=int, default=5, help="unseen ICs for the generalization score")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of the posterior draws, test ICs and NUTS; also the data seed with --data local "
                         "(--data base uses build_problem's own module-level rng, as mcmc_trajectory2.py)")
    ap.add_argument("--n_show", type=int, default=1,
                    help="number of training / unseen ICs to plot, one predictions pdf per IC index "
                         "(n_show > 1: files get the suffix _ic<l>)")
    ap.add_argument("--n_pf", type=int, default=300, help="posterior draws rolled out per IC")
    ap.add_argument("--n_draws", type=int, default=4000, help="draws from the closed-form posteriors")
    ap.add_argument("--substeps", type=int, default=1, help="RK4 substeps in the trajectory likelihood")
    ap.add_argument("--gp_repo", default=os.environ.get("GP_BAYESOPINF_DIR", os.path.join(HERE, "GP-BayesOpInf")))
    ap.add_argument("--gp_restarts", type=int, default=50)
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=list(ALL_METHODS),
                    help="gp = GP-BayesOpInf with their eta = 1e-8; gp_tuned = the same with eta tuned by their "
                         "training-error criterion over --gp_eta_grid (each GP variant fits its own GPs)")
    ap.add_argument("--gp_eta_grid", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                    help="gp_tuned: eta candidates relative to the largest eigenvalue of the GP derivative "
                         "covariance (their 1e-8 is always a candidate too)")
    ap.add_argument("--gp_fallback", action="store_true", 
                    help="if their regularizer search fails (no stable regularizer in the refinement), use the "
                         "regularizer their grid search picked instead of stopping (NOT part of their method; "
                         "reported, and output files get the suffix _gpfallback)")
    ap.add_argument("--data", default="auto", choices=["auto", "base", "local"],
                    help="auto: mcmc_derivative_derv_fixed.build_problem if it imports, else local")
    ap.add_argument("--summarize", action="store_true", help="print a table of all saved runs and exit")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    else:
        run(args)
