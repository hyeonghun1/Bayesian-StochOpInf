"""
Sanity check: MCMC on the deterministic Bayesian OpInf posterior.

Target (Guo, McQuarrie & Willcox, Bayesian OpInf), row-wise:
    likelihood : r_i | o_i ~ N(D o_i, sigma_i^2 I)
    prior      : o_i       ~ N(0, sigma_i^2 Gamma^{-1})
    posterior  : o_i | r_i ~ N(mu_i, Sigma_i)
                 mu_i    = (D^T D + Gamma)^{-1} D^T r_i
                 Sigma_i = sigma_i^2 (D^T D + Gamma)^{-1}

sigma_i^2 is a fixed plug-in value (residual variance of the MAP fit) and
Gamma is fixed, so the target is exactly Gaussian.
NUTS must reproduce (mu_i, Sigma_i) up to Monte Carlo error. If it does not, the bug is in the
log-posterior or the sampler, not in the method.

Need to swap in our own D, R, Gamma, sigma2 in `build_problem` to run this on real data.

Python run in bash:
$ python bayesopinf_mcmc_check.py --noise 0.001 --T_train 3 --T_test 8 --n_pf 500

"""

import os
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin, autocorrelation
from scipy.integrate import solve_ivp
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

numpyro.set_host_device_count(4)
rng = np.random.default_rng(0)

try:                                  # rank-normalized R-hat and bulk/tail ESS (Vehtari et al. 2021)
    import arviz as az                # pip install arviz
except ImportError:
    az = None

CHAIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]    # one color per chain


# 1. Toy problem: a stable quadratic ROM  dq/dt = c + A q + H (q ⊗ q)_compact
r = 3

def compact_kron(q):
    """Non-redundant quadratic terms q_i q_j, i <= j. Works on (r,) or (r, K)."""
    i, j = np.triu_indices(q.shape[0])
    return q[i] * q[j]

def true_operators():
    # linear: damping + rotation
    A = -np.diag([0.4, 0.8, 1.2]) + np.array([[0, 1.0, 0], [-1.0, 0, 0.5], [0, -0.5, 0]])
    c = np.array([0.5, 0.0, -0.3])
    # energy-preserving quadratic: B(q) q with B(q) = sum_k q_k S_k
    # S: skew-symmetric matrices, so that q^T B(q) q = 0 (energy preserving)
    S = [np.array([[0, a, b], [-a, 0, e], [-b, -e, 0]])
         for a, b, e in [(0.6, -0.3, 0.2), (-0.4, 0.5, 0.3), (0.2, 0.1, -0.5)]]
    # convert B(q) q into compact-Kronecker coefficients
    iu, ju = np.triu_indices(r)
    H = np.zeros((r, len(iu)))
    for m, (i, j) in enumerate(zip(iu, ju)):
        e_i, e_j = np.eye(r)[i], np.eye(r)[j]
        # contribution of q_i q_j: S_i e_j + S_j e_i (halved on the diagonal)
        v = S[i] @ e_j + S[j] @ e_i
        H[:, m] = v / 2 if i == j else v
    return c, A, H

def rom_rhs(O):
    c, A, H = O[:, 0], O[:, 1:1 + r], O[:, 1 + r:]
    return lambda t, q: c + A @ q + H @ compact_kron(q)

def operators_to_matrix(c, A, H):
    return np.hstack([c[:, None], A, H])       # O is r x d, row i = o_i

from scipy.signal import savgol_filter, savgol_coeffs

def local_poly_smooth(Q, dt, window=None, poly=3, candidates=range(7, 61, 2)):
    """Local polynomial (Savitzky-Golay) estimates of states and time derivatives.

    Q      : r x K noisy snapshots on a uniform time grid
    window : odd window length in points; None -> chosen by GCV on the states
    poly   : local polynomial degree
    Returns smoothed states, derivative estimates, and the window used.
    """
    K = Q.shape[1]
    if window is None:
        best = np.inf
        for w in candidates:
            if w <= poly or w > K:
                continue
            h0 = savgol_coeffs(w, poly)[w // 2]           # self-weight = diag of hat matrix (interior)
            rss = np.mean((Q - savgol_filter(Q, w, poly, axis=1, mode="interp"))**2)
            gcv = rss / (1.0 - h0)**2
            if gcv < best:
                best, window = gcv, w
    Q_s = savgol_filter(Q, window, poly, axis=1, mode="interp")
    dQ = savgol_filter(Q, window, poly, deriv=1, delta=dt, axis=1, mode="interp")
    return Q_s, dQ, window

def build_problem(noise=0.005, T_train=6.0, K=300, n_traj=5,
                  deriv="lpr", window=None, poly=3, T_test=None):
    """deriv = "fd"  : noisy states + finite differences (original baseline)
       deriv = "lpr" : local-polynomial-smoothed states + derivatives (fair baseline)
       T_test        : if > T_train, also simulate trajectory 0 on (T_train, T_test] with the same dt as
                       held-out test data for time extrapolation. Only [0, T_train] enters D and R,
                       and the training data are identical to T_test=None."""
    c, A, H = true_operators()
    O_true = operators_to_matrix(c, A, H)
    t = np.linspace(0, T_train, K)
    dt = t[1] - t[0]
    n_ext = int(round((T_test - T_train) / dt)) if T_test is not None and T_test > T_train else 0
    t_full = np.concatenate([t, T_train + dt * np.arange(1, n_ext + 1)])     # train grid + test grid
    test_rng = np.random.default_rng(1)          # separate stream: test noise never shifts train data

    Qs_noisy, Qs_fit, Rs, windows = [], [], [], []
    for l in range(n_traj):                      # several ICs -> better identifiability
        q0 = rng.uniform(-1.5, 1.5, r)
        Q_clean = solve_ivp(rom_rhs(O_true), (0, t_full[-1]), q0, t_eval=t_full,
                            rtol=1e-10, atol=1e-12).y
        Q = Q_clean[:, :K] + noise * rng.standard_normal((r, K))
        Qs_noisy.append(Q)
        if l == 0:                               # truth + held-out noisy data for trajectory 0
            q0_true, Q_true = q0, Q_clean
            Q_test = Q_clean[:, K:] + noise * test_rng.standard_normal((r, n_ext))

        if deriv == "fd":
            Q_fit, dQ = Q, np.gradient(Q, t, axis=1, edge_order=2)
        elif deriv == "lpr":
            Q_fit, dQ, w = local_poly_smooth(Q, dt, window=window, poly=poly)
            windows.append(w)
        else:
            raise ValueError(f"unknown deriv = {deriv!r}")
        Qs_fit.append(Q_fit)
        Rs.append(dQ)

    Q_all, R = np.hstack(Qs_fit), np.hstack(Rs)            # r x (n_traj*K)
    K_all = Q_all.shape[1]

    # data matrix D = [1, q^T, (q⊗q)^T]   (K x d), built from the (smoothed) states
    D = np.hstack([np.ones((K_all, 1)), Q_all.T, compact_kron(Q_all).T])
    d = D.shape[1]

    # Tikhonov prior precision (fixed; separate weights for linear / quadratic)
    lam1, lam2 = 1e-2, 1e-1
    Gamma = np.diag([lam1] * (1 + r) + [lam2] * (d - 1 - r))

    # analytical posterior per row
    P = D.T @ D + Gamma
    P_inv = np.linalg.inv(P)
    mu = (P_inv @ D.T @ R.T).T                                   # r x d
    residual = R - mu @ D.T
    sigma2 = np.sum(residual**2, axis=1) / K_all                 # plug-in, fixed
    Sigma = np.stack([s2 * P_inv for s2 in sigma2])              # r x d x d

    info = f", windows = {windows} pts ({windows[0]*dt:.2f} time units)" if windows else ""
    print(f"[{deriv}] d = {d} params/row, K = {K_all} snapshots, "
          f"cond(D^T D + Gamma) = {np.linalg.cond(P):.2e}{info}")
    if n_ext:
        print(f"test window: ({T_train:.1f}, {t_full[-1]:.1f}], {n_ext} held-out snapshots of trajectory 0")
    return dict(t=t, q0=Qs_fit[0][:, 0], Q=Qs_noisy[0], Q_fit=Qs_fit[0],
                D=D, R=R, Gamma=Gamma, sigma2=sigma2, mu=mu, Sigma=Sigma,
                O_true=O_true, deriv=deriv, windows=windows,
                T_train=T_train, t_full=t_full, q0_true=q0_true, Q_true=Q_true, Q_test=Q_test)



# 2. NumPyro model for one row. Written as prior + likelihood (NOT as the
#    closed-form Gaussian), so the check actually tests the log-posterior.
def row_model(D, r_i, Gamma, sigma2_i):
    # Bayesian model for one row: r_i | o ~ N(D o, sigma2_i I), o ~ N(0, sigma2_i Gamma^{-1})
    d = D.shape[1]
    
    # prior: o_i ~ N(0, sigma2_i * inv(Gamma)) 
    prior_cov = sigma2_i * jnp.linalg.inv(Gamma)
    o = numpyro.sample("o", dist.MultivariateNormal(jnp.zeros(d), covariance_matrix=prior_cov))
    
    # likelihood r_i ~ N(Do_i, sigma2 I) with obs=r_i observed.
    # dist.Normal expects a 1D array for the mean and stddev, so we compute Do_i and sqrt(sigma2_i)
    numpyro.sample("r", dist.Normal(D @ o, jnp.sqrt(sigma2_i)), obs=r_i)


# Running NUTS on one row of the problem
def sample_row(prob, i, num_warmup=1000, num_samples=2000, num_chains=4, seed=0,
               return_diagnostics=False):
    # dense_mass=True uses a dense mass matrix (full covariance) during the warmup phase,
    # and uses it as the preconditioner for the NUTS sampler. This is important here since the operators are strongly correlated
    # (D^T D is ill-conditioned) and a diagonal mass matrix only learns per-entry scales, not correlations.
    # Target acceptance probability is set to 0.9 to tune the step size so that 90% of the proposed steps are accepted.
    # Higer target_accept_prob leads to smaller step sizes and more conservative exploration of the posterior,
    # which can be beneficial for complex or high-dimensional posteriors.
    kernel = NUTS(row_model, dense_mass=True, target_accept_prob=0.9)

    # We do `numpyro.set_host_device_count(4)` so we can run in parallel on 4 CPU cores.
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)

    # data for this row: data matrix D, observed data r_i, prior precision Gamma, noise variance sigma2_i
    data = dict(D=jnp.asarray(prob["D"]), r_i=jnp.asarray(prob["R"][i]),
                Gamma=jnp.asarray(prob["Gamma"]), sigma2_i=float(prob["sigma2"][i]))
    key = jax.random.PRNGKey(seed + i)

    if not return_diagnostics:
        mcmc.run(key, **data)
        return np.asarray(mcmc.get_samples(group_by_chain=True)["o"])   # chains x draws x d


    # With diagnostics: run warmup and sampling separately so the warmup draws are kept.
    # extra_fields records, per iteration: divergence flag, number of leapfrog steps (tree size),
    # the current (adapting) step size, and the acceptance probability.
    fields = ("diverging", "num_steps", "adapt_state.step_size", "accept_prob")
    mcmc.warmup(key, collect_warmup=True, extra_fields=fields, **data)
    warm = np.asarray(mcmc.get_samples(group_by_chain=True)["o"])       # chains x num_warmup x d
    warm_f = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
    
    # sampling continues from the adapted state (step size and mass matrix are now frozen)
    mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=fields, **data)
    samples = np.asarray(mcmc.get_samples(group_by_chain=True)["o"])    # chains x num_samples x d
    samp_f = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
    
    return samples, dict(warmup=warm, warmup_fields=warm_f, fields=samp_f)


# 3. Comparison metrics
def gaussian_kl(m0, S0, m1, S1):
    # Kullback-Leibler divergence from N(m0, S0) to N(m1, S1)
    # KL( N(m0,S0) || N(m1,S1) ) = \int (p(x) (\log p(x) - \log q(x))) dx >= 0
    # This is zero only when m0 = m1 and S0 = S1. It is not symmetric: KL(p||q) != KL(q||p) in general.
    # For two multivariate Gaussians, the closed-form formula is:
    # KL(N0 || N1) = 0.5 * (tr(S1^{-1} S0) + (m1 - m0)^T S1^{-1} (m1 - m0) - d + log(det(S1) - log(det(S0))
    #                      ---------------   -----------------------------       -------------------------
    #                          spread                mean offset                        volume 
    #                                        (squared distance between means)
    
    d = len(m0)
    S1_inv = np.linalg.inv(S1)
    dm = m1 - m0
    _, ld0 = np.linalg.slogdet(S0)      # log det(S0)
    _, ld1 = np.linalg.slogdet(S1)      # log det(S1)
    
    KL = 0.5 * (np.trace(S1_inv @ S0) + dm @ S1_inv @ dm - d + ld1 - ld0)
    
    return KL


def compare_row(samples, mu_i, Sigma_i):
    """Compare the MCMC samples to the analytical posterior N(mu_i, Sigma_i) for one row.
    Analytical posterior = N(mu_i, Sigma_i)"""
    # check 1. Did the chains converge, and how much information do they carry?
    ess = np.asarray(effective_sample_size(samples))
    rhat = np.asarray(split_gelman_rubin(samples))
    
    # check 2. Is the mean right?
    flat = samples.reshape(-1, samples.shape[-1])
    m_hat = flat.mean(0)
    S_hat = np.cov(flat, rowvar=False)

    # mean z-scores: should look ~ N(0,1)
    mcse = np.sqrt(np.diag(Sigma_i) / ess)
    z = (m_hat - mu_i) / mcse

    # check 3. Is the covariance right, in every direction?
    # whitened covariance: L^{-1} S_hat L^{-T} should be ~ I
    L = np.linalg.cholesky(Sigma_i)                          # Sigma_i = L L^T
    W = np.linalg.solve(L, np.linalg.solve(L, S_hat).T)      # W = L^{-1} S_hat L^{-T}
    eig = np.linalg.eigvalsh(W)

    rel_frob = np.linalg.norm(S_hat - Sigma_i) / np.linalg.norm(Sigma_i)
    
    kl = gaussian_kl(m_hat, S_hat, mu_i, Sigma_i)
    
    d = len(mu_i)
    # finite-sample baseline: E[KL] ~ d(d+1)/(4n) for n iid draws. NUTS draws
    # are anti-correlated for means (ESS > N) but not for second moments, so use
    # the ESS of the squared whitened coordinates.
    zc = np.linalg.solve(L, (samples - mu_i).reshape(-1, d).T).T.reshape(samples.shape)
    ess2 = np.asarray(effective_sample_size(zc**2)).min()
    kl_expected = d * (d + 1) / (4 * ess2)

    return dict(z=z, ess=ess, ess2=ess2, rhat=rhat, eig=eig, rel_frob=rel_frob,
                kl=kl, kl_expected=kl_expected, m_hat=m_hat, S_hat=S_hat, flat=flat)


# 4. Push-forward: ROM prediction bands from MCMC vs. analytical samples
def pushforward(O_samples, t, q0):
    out = []
    for O in O_samples:
        sol = solve_ivp(rom_rhs(O), (t[0], t[-1]), q0, t_eval=t, rtol=1e-8, atol=1e-10)
        if sol.success and sol.y.shape[1] == len(t):
            out.append(sol.y)
    return np.array(out)                      # n x r x K


# 5. Train / test (time-extrapolation) skill of the push-forward ensemble
def crps_ensemble(Y, y):
    """CRPS of the ensemble Y (n x ...) at the truth y (...), sorted-ensemble formula."""
    n = Y.shape[0]          # number of ensemble members
    X = np.sort(Y, axis=0)
    i = np.arange(1, n + 1).reshape((-1,) + (1,) * y.ndim)
    return 2.0 / n**2 * np.sum((X - y) * (n * (y < X) - i + 0.5), axis=0)


def score_prediction(Y, t, truth, T_train):
    """Score rollouts Y (n x r x K) against the noiseless truth (r x K) separately on the
    training window [0, T_train] and the extrapolation window (T_train, t[-1]].

    rel_err : ||ensemble median - truth||_F / ||truth||_F on the window
    cov     : fraction of (component, time) points where truth lies in the 95% band
    width   : mean width of the 95% band
    crps    : mean CRPS (proper score, same units as q; lower is better)
    """
    lo, med, hi = np.percentile(Y, [2.5, 50, 97.5], axis=0)
    crps = crps_ensemble(Y, truth)
    out = dict(err_t=np.linalg.norm(med - truth, axis=0) / np.linalg.norm(truth, axis=0))
    for name, m in [("train", t <= T_train), ("test", t > T_train)]:
        if m.any():
            out[name] = dict(
                rel_err=np.linalg.norm(med[:, m] - truth[:, m]) / np.linalg.norm(truth[:, m]),
                cov=np.mean((truth[:, m] >= lo[:, m]) & (truth[:, m] <= hi[:, m])),
                width=np.mean(hi[:, m] - lo[:, m]),
                crps=crps[:, m].mean())
    return out


# 3b. Mixing diagnostics
def mixing_summary(samples, diag):
    """Rank-normalized split R-hat and bulk/tail ESS (ArviZ, Vehtari et al. 2021) plus
    sampler statistics. samples: chains x draws x d."""
    out = dict(div=int(diag["fields"]["diverging"].sum()),
               steps=float(diag["fields"]["num_steps"].mean()),
               step_size=float(diag["fields"]["adapt_state.step_size"][:, -1].mean()),
               accept=float(diag["fields"]["accept_prob"].mean()))
    if az is not None:
        ds = az.convert_to_dataset({"o": samples})                  # dims: chain, draw, o_dim_0
        out["rhat_rank"] = np.asarray(az.rhat(ds)["o"])
        out["ess_bulk"] = np.asarray(az.ess(ds, method="bulk")["o"])
        out["ess_tail"] = np.asarray(az.ess(ds, method="tail")["o"])
    return out


def plot_mixing(i, samples, diag, mu_i, Sigma_i, names, path, n_lag=40, n_early=150):
    """One figure per row. For three operator entries (rows of the grid):
         early warmup trace | post-warmup trace | rank plot | autocorrelation
       bottom row: step-size adaptation | tree size per iteration | chain spread | pair plot."""
    n_chain, n_draw, d = samples.shape
    warm = diag["warmup"]
    params = [0, 1 + i, d - 1]                                       # c_i, A_ii, last H entry
    fig, axes = plt.subplots(4, 4, figsize=(15, 12))

    for row, j in enumerate(params):
        m, sd = mu_i[j], np.sqrt(Sigma_i[j, j])
        # (a) start of warmup: the chains travel from their random start into the posterior
        ax = axes[row, 0]
        for c in range(n_chain):
            ax.plot(np.arange(n_early), warm[c, :n_early, j], color=CHAIN_COLORS[c], lw=0.9)
        ax.axhspan(m - 2 * sd, m + 2 * sd, color="0.85", zorder=0)
        ax.set_title(f"{names[j]}: first {n_early} warmup iterations", fontsize=9)
        ax.set_xlabel("iteration", fontsize=8)
        # (b) post-warmup trace: should look like overlapping, stationary "fuzzy caterpillars"
        ax = axes[row, 1]
        for c in range(n_chain):
            ax.plot(samples[c, :, j], color=CHAIN_COLORS[c], lw=0.3, alpha=0.8)
        for k, ls in [(0, "-"), (-2, "--"), (2, "--")]:
            ax.axhline(m + k * sd, color="#1f1f1e", lw=0.8, ls=ls)
        ax.set_title(f"{names[j]}: post-warmup trace (black: exact mean ± 2 sd)", fontsize=9)
        ax.set_xlabel("draw", fontsize=8)
        # (c) rank plot: ranks of each chain's draws within all draws; uniform if chains agree
        ax = axes[row, 2]
        ranks = stats.rankdata(samples[:, :, j].ravel()).reshape(n_chain, n_draw)
        bins = np.linspace(0, n_chain * n_draw, 21)
        for c in range(n_chain):
            h, _ = np.histogram(ranks[c], bins=bins)
            ax.step(bins[:-1], h, where="post", color=CHAIN_COLORS[c], lw=1.2)
        exp = n_draw / 20
        lo_b, hi_b = stats.binom.ppf([0.005, 0.995], n_draw, 1 / 20)   # 99% band for one bin
        ax.axhspan(lo_b, hi_b, color="0.88", zorder=0)
        ax.axhline(exp, color="#1f1f1e", lw=0.8, ls="--")
        ax.set_ylim(0, 2.2 * exp)
        ax.set_title(f"{names[j]}: rank plot (grey: 99% band if uniform)", fontsize=9)
        ax.set_xlabel("rank among all draws", fontsize=8)
        # (d) autocorrelation per chain
        ax = axes[row, 3]
        for c in range(n_chain):
            ac = np.asarray(autocorrelation(samples[c, :, j]))[:n_lag + 1]
            ax.plot(np.arange(n_lag + 1), ac, color=CHAIN_COLORS[c], lw=1.2)
        ax.axhline(0, color="#1f1f1e", lw=0.8)
        ax.set_ylim(-0.6, 1.05)
        ax.set_title(f"{names[j]}: autocorrelation", fontsize=9)
        ax.set_xlabel("lag", fontsize=8)

    # (e) step-size adaptation during warmup (dual averaging), then frozen
    wf, sf = diag["warmup_fields"], diag["fields"]
    n_w = wf["num_steps"].shape[1]
    ax = axes[3, 0]
    for c in range(n_chain):
        ax.semilogy(wf["adapt_state.step_size"][c], color=CHAIN_COLORS[c], lw=0.9)
    ax.set_title("step size during warmup", fontsize=9)
    ax.set_xlabel("warmup iteration", fontsize=8)
    # (f) tree size (leapfrog steps per iteration): drops once the mass matrix is learned
    ax = axes[3, 1]
    for c in range(n_chain):
        steps = np.concatenate([wf["num_steps"][c], sf["num_steps"][c]])
        ax.semilogy(steps, color=CHAIN_COLORS[c], lw=0.4, alpha=0.7)
    ax.axvline(n_w, color="#1f1f1e", lw=0.8, ls=":")
    n_div = int(sf["diverging"].sum())
    ax.set_title("leapfrog steps per iteration (dotted: end of warmup)\n"
                 f"divergences after warmup: {n_div}", fontsize=9)
    ax.set_xlabel("iteration", fontsize=8)
    # (g) per-chain means vs. the exact mean, in units of the exact sd (all d entries)
    ax = axes[3, 2]
    z = (samples.mean(1) - mu_i) / np.sqrt(np.diag(Sigma_i))           # chains x d
    for c in range(n_chain):
        ax.plot(np.arange(d) + (c - 1.5) * 0.12, z[c], "o", ms=4, color=CHAIN_COLORS[c])
    ax.axhline(0, color="#1f1f1e", lw=0.8)
    ax.set_xticks(np.arange(d), names, fontsize=7)
    ax.set_title("per-chain mean − exact mean (in exact sd)", fontsize=9)
    # (h) most correlated pair: draws vs. the exact 95% ellipse
    ax = axes[3, 3]
    corr = Sigma_i / np.sqrt(np.outer(np.diag(Sigma_i), np.diag(Sigma_i)))
    np.fill_diagonal(corr, 0)
    a, b = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    sub = np.random.default_rng(0).choice(n_draw, min(500, n_draw), replace=False)
    for c in range(n_chain):
        ax.plot(samples[c, sub, a], samples[c, sub, b], ".", ms=1.5, color=CHAIN_COLORS[c], alpha=0.5)
    w_, V_ = np.linalg.eigh(Sigma_i[np.ix_([a, b], [a, b])])
    th = np.linspace(0, 2 * np.pi, 200)
    ell = (V_ @ (np.sqrt(w_ * stats.chi2.ppf(0.95, 2))[:, None] * np.vstack([np.cos(th), np.sin(th)])))
    ax.plot(mu_i[a] + ell[0], mu_i[b] + ell[1], color="#1f1f1e", lw=1.2)
    ax.set_xlabel(names[a], fontsize=8); ax.set_ylabel(names[b], fontsize=8)
    ax.set_title(f"most correlated pair (ρ = {corr[a, b]:.2f}); black: exact 95%", fontsize=9)

    handles = [plt.Line2D([], [], color=CHAIN_COLORS[c], lw=2, label=f"chain {c + 1}")
               for c in range(n_chain)]
    fig.suptitle(f"Row {i}: NUTS mixing diagnostics", y=0.995, fontsize=11)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.972), ncol=n_chain,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(path)
    plt.close(fig)





def main(noise=0.005, T_train=6.0, T_test=12.0, K=None, n_pf=500):
    """noise   : std of the snapshot noise
       T_train : fit on [0, T_train]
       T_test  : extrapolate and score on (T_train, T_test]
       K       : training snapshots per trajectory; None keeps the default sampling rate
                 (300 snapshots over 6 time units) whatever T_train is
       n_pf    : number of posterior draws pushed forward through the ROM"""
    K_given = K is not None
    if K is None:
        K = int(round(T_train * 299 / 6.0)) + 1
    # figures go to pdfs/<this file>_pdfs/, tagged with the run settings
    here, stem = os.path.split(os.path.splitext(os.path.abspath(__file__))[0])
    out_dir = os.path.join(here, "pdfs", stem + "_pdfs")
    os.makedirs(out_dir, exist_ok=True)
    tag = (f"noise{noise:g}_Ttrain{T_train:g}_Ttest{T_test:g}" + (f"_K{K}" if K_given else "")
           + (f"_npf{n_pf}" if n_pf != 500 else ""))
    out = lambda name: os.path.join(out_dir, f"{name}_{tag}.pdf")
    print(f"noise = {noise}, train on [0, {T_train}] with K = {K}, test on ({T_train}, {T_test}]")
    prob = build_problem(noise=noise, T_train=T_train, K=K, T_test=T_test)
    r_, d = prob["mu"].shape

    names = ["c"] + [f"A{k}" for k in range(r)] + [f"H{k}" for k in range(d - 1 - r)]
    results = []
    for i in range(r_):
        s, diag = sample_row(prob, i, return_diagnostics=True)
        res = compare_row(s, prob["mu"][i], prob["Sigma"][i])
        results.append(res)
        mix = mixing_summary(s, diag)
        plot_mixing(i, s, diag, prob["mu"][i], prob["Sigma"][i], names, out(f"mixing_row{i}"))
        print(f"\nrow {i}:  min ESS = {res['ess'].min():.0f} (2nd-moment ESS = {res['ess2']:.0f}),  max R-hat = {res['rhat'].max():.3f}")
        print(f"  mean z-scores: max|z| = {np.abs(res['z']).max():.2f}, "
              f"mean z^2 = {np.mean(res['z']**2):.2f}  (expect ~1)")
        tol = 2 * np.sqrt(d / res['ess2'])
        print(f"  whitened cov eigenvalues in [{res['eig'].min():.3f}, {res['eig'].max():.3f}]  "
              f"(expect within ~1 ± {tol:.3f})")
        print(f"  rel. Frobenius cov error = {res['rel_frob']:.3f}")
        print(f"  KL(MCMC || analytic) = {res['kl']:.4f}  (finite-sample baseline ~ {res['kl_expected']:.4f})")
        print(f"  sampler: {mix['div']} divergences after warmup, {mix['steps']:.1f} leapfrog steps/iteration, "
              f"step size {mix['step_size']:.3f}, mean accept prob {mix['accept']:.2f}")
        if "rhat_rank" in mix:
            print(f"  rank-normalized: max R-hat = {mix['rhat_rank'].max():.4f} (want < 1.01), "
                  f"min bulk ESS = {mix['ess_bulk'].min():.0f}, min tail ESS = {mix['ess_tail'].min():.0f} "
                  f"(want > 400)")

    # marginal QQ plots for a few entries of each row
    fig, axes = plt.subplots(r_, 3, figsize=(10, 3 * r_))
    for i, res in enumerate(results):
        for col, j in enumerate([0, 1 + i, d - 1]):
            x = np.sort(res["flat"][:, j])
            p = (np.arange(len(x)) + 0.5) / len(x)
            q_th = stats.norm.ppf(p, prob["mu"][i, j], np.sqrt(prob["Sigma"][i, j, j]))
            ax = axes[i, col]
            ax.plot(q_th, x, ".", ms=1.5, alpha=0.5)
            lo, hi = q_th[0], q_th[-1]
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_title(f"row {i}, {names[j]}", fontsize=9)
            ax.set_xlabel("analytical quantile", fontsize=8)
            ax.set_ylabel("MCMC quantile", fontsize=8)
    fig.tight_layout()
    # fig.savefig("qq_marginals.png", dpi=300)
    fig.savefig(out("qq_marginals"))

    # push-forward bands: training window [0, T_train] + extrapolation window (T_train, T_test]
    idx = rng.choice(results[0]["flat"].shape[0], n_pf, replace=False)
    O_mcmc = np.stack([np.stack([res["flat"][k] for res in results]) for k in idx])
    O_exact = np.stack([
        np.stack([rng.multivariate_normal(prob["mu"][i], prob["Sigma"][i]) for i in range(r_)])
        for _ in range(n_pf)
    ])
    t, T_train = prob["t_full"], prob["T_train"]
    Y_mcmc = pushforward(O_mcmc, t, prob["q0"])
    Y_exact = pushforward(O_exact, t, prob["q0"])
    print(f"\npush-forward: {len(Y_mcmc)}/{n_pf} MCMC and {len(Y_exact)}/{n_pf} analytical rollouts stayed finite")
    for k in range(r_):
        b_m = np.percentile(Y_mcmc[:, k], [2.5, 97.5], axis=0)
        b_e = np.percentile(Y_exact[:, k], [2.5, 97.5], axis=0)
        gap = np.abs(b_m - b_e).max() / (b_e[1] - b_e[0]).max()
        print(f"  q{k}: max band-edge gap / max band width = {gap:.3f}")

    # prediction skill vs. the noiseless truth, training window vs. extrapolation window
    scores = {lab: score_prediction(Y, t, prob["Q_true"], T_train)
              for lab, Y in [("analytical", Y_exact), ("NUTS", Y_mcmc)]}
    hdr = (f"\n{'posterior':>10s} | {'window':>6s} | {'rel err':>7s} | {'95% cov':>7s} | "
           f"{'band width':>10s} | {'CRPS':>7s}")
    print(hdr); print("-" * len(hdr))
    for lab, sc in scores.items():
        for w in ["train", "test"]:
            if w in sc:
                v = sc[w]
                print(f"{lab:>10s} | {w:>6s} | {v['rel_err']:7.4f} | {v['cov']:7.2f} | "
                      f"{v['width']:10.4f} | {v['crps']:7.4f}")

    fig, axes = plt.subplots(r_, 1, figsize=(8, 2.4 * r_), sharex=True)
    for k in range(r_):
        ax = axes[k]
        for Y, lab, col in [(Y_exact, "analytical Gaussian", "C0"), (Y_mcmc, "NUTS", "C1")]:
            lo, med, hi = np.percentile(Y[:, k], [2.5, 50, 97.5], axis=0)
            ax.fill_between(t, lo, hi, color=col, alpha=0.25, label=f"{lab} 95%")
            ax.plot(t, med, color=col, lw=1)
        ax.plot(t, prob["Q_true"][k], "k--", lw=0.8, label="truth")
        ax.plot(prob["t"], prob["Q"][k], "k.", ms=1, label="training data")
        ax.plot(t[t > T_train], prob["Q_test"][k], ".", color="0.6", ms=1, label="test data")
        ax.axvline(T_train, color="0.5", lw=0.8, ls=":")
        ax.set_ylabel(f"$\\hat q_{k}$")
    axes[0].legend(fontsize=7, loc="upper right", ncol=3)
    axes[-1].set_xlabel("t  (dotted: end of training window)")
    fig.tight_layout()
    # fig.savefig("pushforward_bands.png", dpi=300)
    fig.savefig(out("pushforward_bands"))

    # relative error of the median prediction over time: growth past T_train
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for (lab, sc), col in zip(scores.items(), ["C0", "C1"]):
        ax.semilogy(t, sc["err_t"], color=col, lw=1, label=lab)
    ax.axvline(T_train, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("t"); ax.set_ylabel("rel. error of median")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out("extrapolation_error"))
    print(f"saved {{qq_marginals, pushforward_bands, extrapolation_error, mixing_row0-{r_ - 1}}}_{tag}.pdf "
          f"in {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.005, help="std of the snapshot noise")
    ap.add_argument("--T_train", type=float, default=6.0, help="training window [0, T_train]")
    ap.add_argument("--T_test", type=float, default=12.0, help="test window (T_train, T_test]")
    ap.add_argument("--K", type=int, default=None,
                    help="training snapshots per trajectory (default: keep dt = 6/299)")
    ap.add_argument("--n_pf", type=int, default=500,
                    help="posterior draws pushed forward (at most 8000 = 4 chains x 2000)")
    args = ap.parse_args()
    main(noise=args.noise, T_train=args.T_train, T_test=args.T_test, K=args.K, n_pf=args.n_pf)