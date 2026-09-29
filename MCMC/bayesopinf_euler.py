"""
Closed-form Bayesian OpInf vs. NUTS on the 1D compressible Euler example of
Guo, McQuarrie & Willcox (CMAME 2022), using THEIR code for everything upstream of sampling.

Upstream (imported unchanged from github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D,
branch cmame2022: euler1D.py, bayes.py, plot_euler.py):
    FOM         conservative Euler on [0, 2), periodic, nx = 200, first-order backward FD,
                forward Euler, dt = 1e-5, t in [0, 0.03] (3001 snapshots)
    ICs         p = 1e5; periodic cubic splines for rho and u through x = 0, 2/3, 4/3,
                every combination of rho in {20, 24}, u in {95, 105}  -> 64 trajectories
    noise       Gaussian on (rho, rho u, rho e), 5% of fixed variable ranges, then lifted to (u, p, 1/rho)
    ROM         nondimensionalized lifted variables, POD r = 9 from all noisy training snapshots
                (first 1000 per trajectory = t in [0, 0.01)), quadratic-only model dq/dt = H (q (x) q)
    derivatives local-polynomial weights (De Brabanter et al. 2013), k//30 terms
    posterior   o_i ~ N(Ohat_i, sigma_i^2 (D^T D + lambda_i^2 I)^{-1}), with lambda_i from the
                per-row fixed-point update (case 2), lambda_0 = 50, tol 1e-3

This script adds only the NUTS step and the comparison, as in the toy-problem baseline:
    likelihood  r_i | o_i ~ N(D o_i, sigma_i^2 I)
    prior       o_i       ~ N(0, (sigma_i^2 / lambda_i^2) I)
NUTS targets exactly the closed-form posterior, so it should reproduce it up to Monte Carlo error.

Run (needs rom-operator-inference==1.2.1, h5py, ipython; GUO_DIR points at the cloned repo):
    python bayesopinf_euler.py [--r 9] [--noise 0.05] [--testloc 42] [--ndraws 500]
"""

import os
import sys
import time
import itertools
import argparse
import warnings

import numpy as np
np.int = int                      # compat shims for their 2022 code on current numpy
np.warnings = warnings

GUO_DIR = os.environ.get("GUO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "guo"))
sys.path.insert(0, GUO_DIR)
os.makedirs("/storage1/euler1D", exist_ok=True)      # their config.py insists this folder exists
import euler1D                    # noqa: E402
import bayes                      # noqa: E402
import plot_euler                 # noqa: E402

import jax                        # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp           # noqa: E402
import numpyro                    # noqa: E402
import numpyro.distributions as dist   # noqa: E402
from numpyro.infer import MCMC, NUTS   # noqa: E402
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin   # noqa: E402
import scipy.linalg as la         # noqa: E402
from scipy import stats           # noqa: E402
import matplotlib                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

numpyro.set_host_device_count(4)


# 1. Their data generation and posterior construction ------------------------
def generate_data(seed):
    """Same as plot_euler.generate_training_data(), without saving. Their code does not seed
    the noise; we seed numpy's global RNG for reproducibility."""
    np.random.seed(seed)
    solver = euler1D.EulerROMSolver(nx=200, nt=3000, L=2, tf=3e-2, gamma=1.4)   # config.EULER_DOMAIN
    rho, u = [20, 24], [95, 105]
    t0 = time.time()
    for init_params in itertools.product(rho, rho, rho, u, u, u):
        solver.add_snapshot_set(init_params, plot_init=False)
    print(f"FOM: 64 trajectories, n = {3*solver.n}, k = {solver.k}, {time.time()-t0:.1f}s")
    return solver


def bayesopinf_posterior(solver, trainsize, level, r, reg0=50, tol=1e-3):
    """Body of plot_euler.generate_bayesopinf_results() up to the posterior (no file I/O)."""
    rom = solver.train_rom(r, noise_level=level, ktrain=trainsize, reg=reg0)
    post, reg = bayes.construct_posterior(rom, rom.reg, case=2)
    Q_ = rom._training_states_
    R_ = rom.solver_.B.T
    D = rom.solver_.A
    gs = la.eigvalsh(D.T @ D)
    rom.fit(rom.Vr, Q_, R_, P=reg)
    for it in range(50):
        post, regnew = bayes.construct_posterior(rom, reg, case=2, gs=gs)
        diff = la.norm(regnew - reg) / la.norm(reg)
        if diff < tol:
            break
        reg = regnew
    print(f"lambda fixed-point: {it+1} iterations (final diff {diff:.1e})")
    return rom, post, np.asarray(reg), D, R_


# 2. NUTS on one row (prior + likelihood, not the closed form) ---------------
def row_model(D, r_i, prior_sd, sigma_i):
    d = D.shape[1]
    o = numpyro.sample("o", dist.Normal(jnp.zeros(d), prior_sd).to_event(1))
    numpyro.sample("r", dist.Normal(D @ o, sigma_i), obs=r_i)


def sample_row(D, r_i, lam_i, sigma2_i, num_warmup=1000, num_samples=2000, num_chains=4, seed=0):
    kernel = NUTS(row_model, dense_mass=True, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), D=jnp.asarray(D), r_i=jnp.asarray(r_i),
             prior_sd=float(np.sqrt(sigma2_i) / lam_i), sigma_i=float(np.sqrt(sigma2_i)),
             extra_fields=("diverging",))
    return (np.asarray(mcmc.get_samples(group_by_chain=True)["o"]),
            int(np.sum(mcmc.get_extra_fields()["diverging"])))


# 3. Comparison metrics (same as the baseline) ------------------------------
def gaussian_kl(m0, S0, m1, S1):
    d = len(m0)
    S1_inv = np.linalg.inv(S1)
    dm = m1 - m0
    _, ld0 = np.linalg.slogdet(S0)
    _, ld1 = np.linalg.slogdet(S1)
    return 0.5 * (np.trace(S1_inv @ S0) + dm @ S1_inv @ dm - d + ld1 - ld0)


def compare_row(samples, mu_i, Sigma_i):
    ess = np.asarray(effective_sample_size(samples))
    rhat = np.asarray(split_gelman_rubin(samples))
    flat = samples.reshape(-1, samples.shape[-1])
    m_hat, S_hat = flat.mean(0), np.cov(flat, rowvar=False)
    z = (m_hat - mu_i) / np.sqrt(np.diag(Sigma_i) / ess)
    L = np.linalg.cholesky(Sigma_i)
    W = np.linalg.solve(L, np.linalg.solve(L, S_hat).T)
    d = len(mu_i)
    zc = np.linalg.solve(L, (samples - mu_i).reshape(-1, d).T).T.reshape(samples.shape)
    ess2 = np.asarray(effective_sample_size(zc**2)).min()
    return dict(z=z, ess=ess, ess2=ess2, rhat=rhat, eig=np.linalg.eigvalsh(W),
                rel_frob=np.linalg.norm(S_hat - Sigma_i) / np.linalg.norm(Sigma_i),
                kl=gaussian_kl(m_hat, S_hat, mu_i, Sigma_i), kl_expected=d * (d + 1) / (4 * ess2),
                flat=flat)


# 4. Push-forward with their integrator (RK45, as in bayes.simulate_posterior) --
def pushforward(post, O_samples, q0_, t):
    out = []
    for O in O_samples:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q = post._construct_rom(O).predict(q0_, t, None, method="RK45")
        if q.shape[1] == t.shape[0] and np.all(np.isfinite(q)):
            out.append(q)
    return np.array(out)


def crps_ensemble(Y, y):
    n = Y.shape[0]
    X = np.sort(Y, axis=0)
    i = np.arange(1, n + 1).reshape((-1,) + (1,) * y.ndim)
    return 2.0 / n**2 * np.sum((X - y) * (n * (y < X) - i + 0.5), axis=0)


def score_prediction(Y, truth, ktrain):
    """Reduced-space scores vs. the noise-free projected FOM, split at the end of training."""
    lo, med, hi = np.percentile(Y, [2.5, 50, 97.5], axis=0)
    crps = crps_ensemble(Y, truth)
    out = dict(err_t=np.linalg.norm(med - truth, axis=0) / np.linalg.norm(truth, axis=0), med=med)
    for name, m in [("train", slice(0, ktrain)), ("test", slice(ktrain, None))]:
        out[name] = dict(
            rel_err=np.linalg.norm(med[:, m] - truth[:, m]) / np.linalg.norm(truth[:, m]),
            cov=np.mean((truth[:, m] >= lo[:, m]) & (truth[:, m] <= hi[:, m])),
            width=np.mean(hi[:, m] - lo[:, m]), crps=crps[:, m].mean())
    return out


# ---------------------------------------------------------------------------
def main(a):
    rng = np.random.default_rng(a.seed)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs", "bayesopinf_euler_pdfs")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"r{a.r}_noise{a.noise:g}_ic{a.testloc}"
    out = lambda name: os.path.join(out_dir, f"{name}_{tag}.pdf")

    solver = generate_data(a.seed)
    rom, post, lam, D, R_ = bayesopinf_posterior(solver, a.trainsize, a.noise, a.r)
    r_, d = post.means.shape
    DTD = bayes._symmetrize(D.T @ D)
    Sigmas = np.array([la.inv(S) for S in post.invcovariances])
    # recover sigma_i^2 exactly as their code uses it: invcov_i = (D^T D + lam_i^2 I) / sigma_i^2
    sigma2 = np.array([np.mean(np.diag(DTD + l**2 * np.eye(d)) / np.diag(Sinv))
                       for l, Sinv in zip(lam, post.invcovariances)])
    mu_chk = np.array([la.solve(DTD + l**2 * np.eye(d), D.T @ R_[:, i]) for i, l in enumerate(lam)])
    print(f"D: {D.shape}, cond(D^T D) = {np.linalg.cond(DTD):.2e}, "
          f"lambda = {np.array2string(lam, precision=2)}")
    print(f"check: |their mean - ridge solve| / |mean| = "
          f"{np.linalg.norm(mu_chk - post.means)/np.linalg.norm(post.means):.1e}")

    results = []
    for i in range(r_):
        t0 = time.time()
        s, ndiv = sample_row(D, R_[:, i], lam[i], sigma2[i], seed=a.seed + i)
        res = compare_row(s, post.means[i], Sigmas[i])
        results.append(res)
        tol = 2 * np.sqrt(d / res["ess2"])
        print(f"\nrow {i} ({time.time()-t0:.0f}s, {ndiv} divergences): min ESS = {res['ess'].min():.0f} "
              f"(2nd-moment ESS = {res['ess2']:.0f}), max R-hat = {res['rhat'].max():.3f}")
        print(f"  mean z-scores: max|z| = {np.abs(res['z']).max():.2f}, mean z^2 = {np.mean(res['z']**2):.2f}  (expect ~1)")
        print(f"  whitened cov eigenvalues in [{res['eig'].min():.3f}, {res['eig'].max():.3f}]  (expect ~1 ± {tol:.3f})")
        print(f"  rel. Frobenius cov error = {res['rel_frob']:.3f}")
        print(f"  KL(MCMC || analytic) = {res['kl']:.4f}  (finite-sample baseline ~ {res['kl_expected']:.4f})")

    # QQ plots: first, a middle and last quadratic coefficient of each row
    fig, axes = plt.subplots(r_, 3, figsize=(10, 2.3 * r_))
    for i, res in enumerate(results):
        for col, j in enumerate([0, d // 2, d - 1]):
            x = np.sort(res["flat"][:, j])
            p = (np.arange(len(x)) + 0.5) / len(x)
            q_th = stats.norm.ppf(p, post.means[i, j], np.sqrt(Sigmas[i, j, j]))
            ax = axes[i, col]
            ax.plot(q_th, x, ".", ms=1.5, alpha=0.5)
            ax.plot([q_th[0], q_th[-1]], [q_th[0], q_th[-1]], "k--", lw=1)
            ax.set_title(f"row {i}, H[{j}]", fontsize=9)
    fig.tight_layout(); fig.savefig(out("qq_marginals"))

    # push-forward for one test IC, from its noisy initial condition (as in their code)
    t = solver.t
    snaps_noised = solver.apply_noise(a.noise)[a.testloc]
    q0_ = rom.Vr.T @ solver.nondimensionalize(snaps_noised[:, 0])
    truth_ = rom.Vr.T @ solver.nondimensionalize(solver.snapshots[a.testloc])
    idx = rng.choice(results[0]["flat"].shape[0], a.ndraws, replace=False)
    O_mcmc = np.stack([np.stack([res["flat"][k] for res in results]) for k in idx])
    O_exact = np.stack([post._sample_operator_matrix() for _ in range(a.ndraws)])
    t0 = time.time()
    Y_exact = pushforward(post, O_exact, q0_, t)
    Y_mcmc = pushforward(post, O_mcmc, q0_, t)
    print(f"\npush-forward (IC {a.testloc}, {time.time()-t0:.0f}s): {len(Y_mcmc)}/{a.ndraws} NUTS and "
          f"{len(Y_exact)}/{a.ndraws} analytical rollouts completed")
    for k in range(r_):
        b_m = np.percentile(Y_mcmc[:, k], [2.5, 97.5], axis=0)
        b_e = np.percentile(Y_exact[:, k], [2.5, 97.5], axis=0)
        print(f"  q{k}: max band-edge gap / max band width = {np.abs(b_m - b_e).max() / (b_e[1] - b_e[0]).max():.3f}")

    scores = {lab: score_prediction(Y, truth_, a.trainsize)
              for lab, Y in [("analytical", Y_exact), ("NUTS", Y_mcmc)]}
    hdr = (f"\n{'posterior':>10s} | {'window':>6s} | {'rel err':>7s} | {'95% cov':>7s} | "
           f"{'band width':>10s} | {'CRPS':>7s} | {'full-state err (theirs)':>23s}")
    print(hdr); print("-" * len(hdr))
    for lab, sc in scores.items():
        full = solver.redimensionalize(rom.Vr @ sc["med"])
        e_tr, e_te = plot_euler._train_predict_error(solver, a.trainsize, full, solver.snapshots[a.testloc])
        for w, e in [("train", e_tr), ("test", e_te)]:
            v = sc[w]
            print(f"{lab:>10s} | {w:>6s} | {v['rel_err']:7.4f} | {v['cov']:7.2f} | "
                  f"{v['width']:10.4f} | {v['crps']:7.4f} | {e:23.4f}")

    n_show = 4
    fig, axes = plt.subplots(n_show, 1, figsize=(8, 2.3 * n_show), sharex=True)
    Q_noisy = rom.Vr.T @ solver.nondimensionalize(snaps_noised)
    for k in range(n_show):
        ax = axes[k]
        for Y, lab, col in [(Y_exact, "analytical Gaussian", "C0"), (Y_mcmc, "NUTS", "C1")]:
            lo, med, hi = np.percentile(Y[:, k], [2.5, 50, 97.5], axis=0)
            ax.fill_between(t, lo, hi, color=col, alpha=0.25, label=f"{lab} 95%")
            ax.plot(t, med, color=col, lw=1)
        ax.plot(t, truth_[k], "k--", lw=0.8, label="projected FOM")
        ax.plot(t[:a.trainsize], Q_noisy[k, :a.trainsize], "k.", ms=0.5, label="training data")
        ax.axvline(t[a.trainsize], color="0.5", lw=0.8, ls=":")
        ax.set_ylabel(f"$\\hat q_{k}$")
    axes[0].legend(fontsize=7, loc="upper right", ncol=2)
    axes[-1].set_xlabel("t  (dotted: end of training window)")
    fig.tight_layout(); fig.savefig(out("pushforward_bands"))

    fig, ax = plt.subplots(figsize=(7, 2.6))
    for (lab, sc), col in zip(scores.items(), ["C0", "C1"]):
        ax.semilogy(t, sc["err_t"], color=col, lw=1, label=lab)
    ax.axvline(t[a.trainsize], color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("t"); ax.set_ylabel("rel. error of median"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out("extrapolation_error"))
    print(f"saved {{qq_marginals, pushforward_bands, extrapolation_error}}_{tag}.pdf in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=9)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--trainsize", type=int, default=1000)
    ap.add_argument("--testloc", type=int, default=42)
    ap.add_argument("--ndraws", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
