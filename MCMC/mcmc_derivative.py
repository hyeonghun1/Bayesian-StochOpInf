"""
Sanity check: MCMC on the deterministic Bayesian OpInf posterior.

Target (Guo, McQuarrie & Willcox, Bayesian OpInf), row-wise:
    likelihood : r_i | o_i ~ N(D o_i, sigma_i^2 I)
    prior      : o_i       ~ N(0, sigma_i^2 Gamma^{-1})
    posterior  : o_i | r_i ~ N(mu_i, Sigma_i)
                 mu_i    = (D^T D + Gamma)^{-1} D^T r_i
                 Sigma_i = sigma_i^2 (D^T D + Gamma)^{-1}

Framework as in Guo et al.:
  - D = [1, q^T, (q kron q)^T] from the RAW noisy snapshots; only the derivative target R is estimated
    (default: De Brabanter weighted difference quotients with K // 30 terms, their ddt_localpoly).
  - Gamma_i = lam_i^2 I per row, lam_i from their Bayesian fixed-point iteration (Algorithm 2,
    case 2 of bayes.construct_posterior) started at lam = 50, tol 1e-3, at most 50 iterations.
  - sigma_i^2 = (||D mu_i - r_i||^2 + lam_i^2 ||mu_i||^2) / K, as in their code.
Once lam_i and sigma_i^2 are fixed, the target is exactly Gaussian.
NUTS must reproduce (mu_i, Sigma_i) up to Monte Carlo error. If it does not, the bug is in the
log-posterior or the sampler, not in the method.

Need to swap in our own D, R, Gamma, sigma2 in `build_problem` to run this on real data.

Prior (--prior):
  tikhonov  (default) Gamma_i = lam_i^2 I with lam_i from the fixed point, or the fixed --lam (as above).
  wide           lam = 1e-4, i.e. Gamma_i = 1e-8 I: a proper Gaussian with prior sd = 1e4 * sigma_i per entry.
  flat           lam = 0, i.e. Gamma_i = 0: improper flat prior, NumPyro model without any prior term.
                 The posterior is the OLS posterior N((D^T D)^{-1} D^T r_i, sigma_i^2 (D^T D)^{-1}),
                 proper as long as D has full column rank. sigma_i^2 = ||D mu_i - r_i||^2 / K.
  uniform        o_ij ~ U(-b, b) independently, b = --uniform_bound (default 10): a proper, bounded
                 uniform prior. The posterior is the OLS posterior truncated to the box [-b, b]^d; when
                 the box does not cut into it (checked and printed), it equals the flat-prior posterior,
                 which is what the analytical reference uses.
  With wide/flat/uniform the lam fixed point is skipped and --lam is ignored.

Data-generating system: --system original | weak_damping | limit_cycle (defined in toy_systems.py).

Usage:
    python mcmc_derivactive.py                                   # Guo et al. setup (dbr + lam fixed point)
    python mcmc_derivactive.py --noise 0.05 --T_train 6 --T_test 12 --K 300 --n_pf 100 --system limit_cycle
    python mcmc_derivactive.py --deriv lpr --lam 0.316           # old setup: SG derivatives, Gamma = 0.1 I
    python mcmc_derivactive.py --dbr_edge trim                   # drop De Brabanter boundary snapshots
    python mcmc_derivactive.py --prior flat                      # noninformative (flat) prior
    python mcmc_derivactive.py --prior wide                      # noninformative, proper Gaussian
    python mcmc_derivactive.py --prior uniform --uniform_bound 10  # proper uniform U(-10, 10) per entry
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

numpyro.set_host_device_count(4)      # Runs 4 chains in parallel, one per CPU core.
rng = np.random.default_rng(0)

try:                                  # rank-normalized R-hat and bulk/tail ESS (Vehtari et al. 2021)
    import arviz as az                # pip install arviz
except ImportError:
    az = None

CHAIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]    # one color per chain


# 1. Toy problem: the data-generating quadratic ROM  dq/dt = c + A q + H (q kron q)_compact.
#    The systems (original, weak_damping, limit_cycle) live in toy_systems.py.
from toy_systems import r, compact_kron, true_operators, rom_rhs, operators_to_matrix, SYSTEMS

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


def debrabanter_derivative(Q, dt, k=None):
    """First derivative by De Brabanter et al. (JMLR 2013) weighted symmetric difference
    quotients, implemented exactly as Guo et al.'s `ddt_localpoly` (euler1D.py, cmame2022):

        interior  (k <= i < K-k): dq_i = sum_{j=1}^{k} w_j (q_{i+j} - q_{i-j}) / (2 j dt),
                                  w_j = 6 j^2 / (k (k+1) (2k+1))
        boundary  (1 <= i < k and mirrored): the first i weights, renormalized to sum to 1
        end rows  i = 0 and i = K-1: left at ZERO (as in their matrix W)

    Q : r x K noisy snapshots,  k : number of terms (default K // 30, as in their code)
    Returns dQ (r x K) and k. The states themselves are not smoothed.
    """
    K = Q.shape[1]
    if k is None:
        k = K // 30
    j = np.arange(1, k + 1)
    w = 6.0 * j**2 / (k * (k + 1) * (2 * k + 1))
    dts = 2 * j * dt
    W = np.zeros((K, K))
    coeffs = w / dts
    row = np.concatenate([-coeffs[::-1], [0.0], coeffs])
    for i in range(k, K - k):                         # interior rows
        W[i, i - k:i + k + 1] = row
    for i in range(1, k):                             # boundary rows
        wi = w[:i] / w[:i].sum()
        coeffs = wi / dts[:i]
        row_i = np.concatenate([-coeffs[::-1], [0.0], coeffs])
        wlen = 2 * i + 1
        W[i, :wlen] = row_i
        W[K - i - 1, -wlen:] = row_i
    return Q @ W.T, k


def guo_posterior_params(D, R, lams):
    """Per-row quantities of Guo et al.'s posterior at regularization lams (r,), where the
    Tikhonov penalty of row i is lams[i]^2 ||o_i||^2 (their convention: reg = lambda, Lambda = lambda^2 I).
    Mirrors bayes.construct_posterior (case 2):
        mu_i     = (D^T D + lam_i^2 I)^{-1} D^T r_i
        sigma2_i = (||D mu_i - r_i||^2 + lam_i^2 ||mu_i||^2) / K      (opinf solver.residual / trainsize)
        gamma_i  = sum_k g_k / (lam_i^2 + g_k),  g = eig(D^T D)
        lam_new_i = sqrt(gamma_i sigma2_i / ||mu_i||^2)
    """
    K, d = D.shape
    G = D.T @ D
    G = 0.5 * (G + G.T)                                # their _symmetrize
    gs = np.linalg.eigvalsh(G)
    lam2 = np.asarray(lams, dtype=float)**2
    mu = np.stack([np.linalg.solve(G + l2 * np.eye(d), D.T @ R[i]) for i, l2 in enumerate(lam2)])
    misfit = np.sum((mu @ D.T - R)**2, axis=1)
    sigma2 = (misfit + lam2 * np.sum(mu**2, axis=1)) / K
    gamma = np.sum(gs[None, :] / (lam2[:, None] + gs[None, :]), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_new = np.sqrt(gamma * sigma2 / np.sum(mu**2, axis=1))
    return mu, sigma2, lam_new


def guo_regularization(D, R, reg0=50.0, tol=1e-3, max_iter=50):
    """Guo et al.'s Bayesian fixed-point iteration for the per-row regularization
    (plot_euler.py, cmame2022; case 2 of bayes.construct_posterior):

        reg  <- update(reg0 for every row)                 # first call with scalar reg0
        repeat at most 50 times:
            regnew <- update(reg)
            diff = ||regnew - reg|| / ||reg||              # joint over all rows
            if diff < tol (1e-3): stop                     # posterior uses reg, NOT regnew
            reg <- regnew

    Returns reg (r,) as lambda (penalty lambda^2), sigma2 (r,) at reg, #iterations, converged flag.
    A row whose mean shrinks to zero sends lambda -> inf (non-finite update); that is reported as
    not converged, as happens in their code for some noise draws.
    """
    r = R.shape[0]
    _, _, reg = guo_posterior_params(D, R, np.full(r, float(reg0)))
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        if not np.all(np.isfinite(reg)):
            break
        _, sigma2, regnew = guo_posterior_params(D, R, reg)
        diff = np.linalg.norm(regnew - reg) / np.linalg.norm(reg)
        if diff < tol:
            converged = True
            break
        reg = regnew
    _, sigma2, _ = guo_posterior_params(D, R, reg)
    return reg, sigma2, it, converged

def build_problem(noise=0.005, T_train=6.0, K=300, n_traj=20,
                  deriv="dbr", window=None, poly=3, k_dbr=None, dbr_edge="guo", lam=None, lam0=50.0,
                  tol=1e-3, T_test=None, system="original", prior="tikhonov", uniform_bound=10.0):
    """Guo et al. framework: D is always built from the RAW noisy states; only the derivative
    target R is estimated. Row i has the Tikhonov penalty lam_i^2 ||o_i||^2, i.e. the prior
    precision Gamma_i = lam_i^2 I (their convention: lam is the square root of the penalty weight).

       deriv = "dbr" : De Brabanter weighted difference quotients (their ddt_localpoly; default)
       deriv = "lpr" : Savitzky-Golay local-polynomial derivatives (GCV window)
       deriv = "fd"  : second-order finite differences
       dbr_edge : "guo"  -> keep every snapshot, exactly as their code (reduced stencils in the first
                            and last k points; the very first and last derivative are ZERO)
                  "trim" -> drop the first and last k snapshots of every trajectory from D and R
       prior : "tikhonov" -> Gamma_i = lam_i^2 I with lam as below (default)
               "wide" -> lam = 1e-4 (Gamma_i = 1e-8 I, proper Gaussian, prior sd = 1e4 sigma_i); lam ignored
               "flat" -> lam = 0 (Gamma_i = 0, improper flat prior, posterior = OLS posterior); lam ignored
               "uniform" -> o_ij ~ U(-uniform_bound, uniform_bound); analytical reference = OLS posterior
                            (exact when the box does not truncate it); lam ignored
       prior : "tikhonov" -> Gamma_i = lam_i^2 I with lam as below (default)
               "wide" -> lam = 1e-4 (Gamma_i = 1e-8 I, proper Gaussian, prior sd = 1e4 sigma_i); lam ignored
               "flat" -> lam = 0 (Gamma_i = 0, improper flat prior, posterior = OLS posterior); lam ignored
               "uniform" -> o_ij ~ U(-uniform_bound, uniform_bound); analytical reference = OLS posterior
                            (exact when the box does not truncate it); lam ignored
       lam   : (prior = "tikhonov" only) None -> per-row lam_i by their fixed-point iteration from lam0 = 50,
               tol = 1e-3, at most 50 iterations (Algorithm 2 / case 2);
               a number -> the same fixed lam for every row (penalty lam^2)
       noise         : relative snapshot noise: std of variable j = noise * range (max - min) of
                       variable j over the clean training snapshots of all trajectories (0.05 = 5%)
       T_test        : if > T_train, also simulate trajectory 0 on (T_train, T_test] with the same dt as
                       held-out test data for time extrapolation. Only [0, T_train] enters D and R,
                       and the training data are identical to T_test=None."""
    if prior == "wide":
        lam = 1e-4                               # Gamma = 1e-8 I
    elif prior in ("flat", "uniform"):
        lam = 0.0                                # Gamma = 0 (uniform: box constraint handled in row_model)
    elif prior != "tikhonov":
        raise ValueError(f"unknown prior = {prior!r}")
    c, A, H = true_operators(system)
    O_true = operators_to_matrix(c, A, H)
    t = np.linspace(0, T_train, K)
    dt = t[1] - t[0]
    n_ext = int(round((T_test - T_train) / dt)) if T_test is not None and T_test > T_train else 0
    t_full = np.concatenate([t, T_train + dt * np.arange(1, n_ext + 1)])     # train grid + test grid
    test_rng = np.random.default_rng(1)          # separate stream: test noise never shifts train data

    # clean trajectories and unit noise draws first (same rng order as adding the noise in place), so the
    # noise level can depend on the clean data
    Qs_clean, Zs = [], []
    for l in range(n_traj):                      # several ICs -> better identifiability
        q0 = rng.uniform(-1.5, 1.5, r)
        Q_clean = solve_ivp(rom_rhs(O_true), (0, t_full[-1]), q0, t_eval=t_full,
                            rtol=1e-10, atol=1e-12).y
        Qs_clean.append(Q_clean)
        Zs.append(rng.standard_normal((r, K)))
        if l == 0:                               # truth + held-out noisy data for trajectory 0
            q0_true, Q_true = q0, Q_clean
            Z_test = test_rng.standard_normal((r, n_ext))
    # per variable: noise x its range over the clean training data
    Q_train = np.hstack([Qc[:, :K] for Qc in Qs_clean])
    noise_std = noise * (Q_train.max(axis=1) - Q_train.min(axis=1))
    print(f"noise = {100 * noise:g}% of each variable's range: std = {np.array2string(noise_std, precision=4)}")
    Q_test = Q_true[:, K:] + noise_std[:, None] * Z_test

    Qs_noisy, Qs_D, Rs, windows = [], [], [], []
    for Q_clean, Z in zip(Qs_clean, Zs):
        Q = Q_clean[:, :K] + noise_std[:, None] * Z
        Qs_noisy.append(Q)

        keep = slice(0, K)                       # snapshots of this trajectory that enter D and R
        if deriv == "dbr":
            dQ, kk = debrabanter_derivative(Q, dt, k=k_dbr)
            windows.append(kk)
            if dbr_edge == "trim":
                keep = slice(kk, K - kk)
            elif dbr_edge != "guo":
                raise ValueError(f"unknown dbr_edge = {dbr_edge!r}")
        elif deriv == "fd":
            dQ = np.gradient(Q, t, axis=1, edge_order=2)
        elif deriv == "lpr":
            _, dQ, w = local_poly_smooth(Q, dt, window=window, poly=poly)   # smoothed states unused
            windows.append(w)
        else:
            raise ValueError(f"unknown deriv = {deriv!r}")
        Qs_D.append(Q[:, keep])
        Rs.append(dQ[:, keep])

    # D from the raw noisy states, R from the derivative estimates
    Q_all, R = np.hstack(Qs_D), np.hstack(Rs)                    # r x (sum of kept snapshots)
    K_all = Q_all.shape[1]

    # data matrix D = [1, q^T, (q⊗q)^T]   (K x d), built from the raw noisy states
    D = np.hstack([np.ones((K_all, 1)), Q_all.T, compact_kron(Q_all).T])
    d = D.shape[1]

    # per-row regularization lam_i (penalty lam_i^2) and plug-in noise variance sigma2_i
    if lam is None:
        lams, sigma2, n_it, converged = guo_regularization(D, R, reg0=lam0, tol=tol)
        status = f"fixed point {'converged' if converged else 'NOT converged'} after {n_it} iterations"
    else:
        lams = np.full(r, float(lam))
        _, sigma2, _ = guo_posterior_params(D, R, lams)
        status = {"tikhonov": "fixed lam", "wide": "noninformative Gaussian prior (lam = 1e-4, Gamma = 1e-8 I)",
                  "flat": "noninformative flat (improper uniform) prior (lam = 0, Gamma = 0)",
                  "uniform": f"proper uniform prior U(-{uniform_bound:g}, {uniform_bound:g}) on every entry"}[prior]

    # analytical posterior per row: mu_i = (G + lam_i^2 I)^{-1} D^T r_i, Sigma_i = sigma2_i (G + lam_i^2 I)^{-1}
    G = D.T @ D
    G = 0.5 * (G + G.T)
    Gamma = np.stack([l**2 * np.eye(d) for l in lams])           # r x d x d prior precision / sigma2
    P_inv = np.stack([np.linalg.inv(G + Gi) for Gi in Gamma])    # r x d x d
    mu = np.stack([P_inv[i] @ D.T @ R[i] for i in range(r)])     # r x d
    Sigma = sigma2[:, None, None] * P_inv                        # r x d x d

    unit = "terms" if deriv == "dbr" else "pts"
    info = f", {deriv} window = {windows[0]} {unit}" if windows else ""
    print(f"[{deriv}] d = {d} params/row, K = {K_all} snapshots, cond(D^T D) = {np.linalg.cond(G):.2e}{info}")
    print(f"  regularization: {status}")
    for i in range(r):
        print(f"  row {i}: lam = {lams[i]:.4g} (penalty lam^2 = {lams[i]**2:.4g}), sigma2 = {sigma2[i]:.3e}, "
              f"cond(D^T D + lam^2 I) = {np.linalg.cond(G + Gamma[i]):.2e}")
    if prior == "uniform":
        # distance from the posterior mean to the nearest box face, in posterior sd (per row)
        sd = np.sqrt(np.stack([np.diag(S) for S in Sigma]))
        margin = ((uniform_bound - np.abs(mu)) / sd).min(axis=1)
        print(f"  uniform box: nearest face at {np.array2string(margin, precision=1)} posterior sd per row "
              f"(> ~5: truncation negligible, analytical reference exact)")
        if np.any(margin < 5):
            print("  WARNING: the uniform box truncates the posterior; MCMC will NOT match the Gaussian reference")
    if lam is None and not converged:
        print("  WARNING: the lam fixed point did not converge; the posterior uses the last iterate")
    if n_ext:
        print(f"test window: ({T_train:.1f}, {t_full[-1]:.1f}], {n_ext} held-out snapshots of trajectory 0")
    # ROM initial condition = first raw noisy snapshot (as in their plot_euler.py: init_ = Vr^T noisy[:, 0])
    return dict(t=t, q0=Qs_noisy[0][:, 0], Q=Qs_noisy[0],
                Qs=np.stack(Qs_noisy),                               # n_traj x r x K noisy snapshots
                D=D, R=R, Gamma=Gamma, lam=lams, sigma2=sigma2, mu=mu, Sigma=Sigma,
                O_true=O_true, deriv=deriv, windows=windows, prior=prior, uniform_bound=uniform_bound,
                T_train=T_train, t_full=t_full, q0_true=q0_true, Q_true=Q_true, Q_test=Q_test,
                noise_std=noise_std)



# 2. NumPyro model for one row. Written as prior + likelihood (NOT as the
#    closed-form Gaussian), so the check actually tests the log-posterior.
def row_model(D, r_i, Gamma, sigma2_i, flat=False, bound=None):
    # Bayesian model for one row: r_i | o ~ N(D o, sigma2_i I), o ~ N(0, sigma2_i Gamma^{-1})
    # flat=True: Gamma = 0 has no inverse, so the prior is an improper uniform density on R^d
    # (no prior term in the log-density); the likelihood alone defines the posterior.
    # bound=b: proper uniform prior o_j ~ U(-b, b) independently. Sampled as u ~ U(-1, 1)^d, o = b u
    # (constant Jacobian, same posterior). The fixed U(-1, 1) keeps the box transform free of traced
    # arguments; U(-b, b) with a traced b breaks NumPyro's separate warmup()/run() calls. NUTS moves
    # in logit(u) space, so the box constraint is handled exactly.
    d = D.shape[1]

    if bound is not None:
        u = numpyro.sample("u", dist.Uniform(-jnp.ones(d), jnp.ones(d)).to_event(1))
        o = numpyro.deterministic("o", bound * u)
    elif flat:
        o = numpyro.sample("o", dist.ImproperUniform(dist.constraints.real_vector, (), (d,)))
    else:
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

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)

    # data for this row: data matrix D, observed data r_i, prior precision Gamma_i = lam_i I, noise variance sigma2_i
    data = dict(D=jnp.asarray(prob["D"]), r_i=jnp.asarray(prob["R"][i]),
                Gamma=jnp.asarray(prob["Gamma"][i]), sigma2_i=float(prob["sigma2"][i]),
                flat=bool(prob["prior"] == "flat"),
                bound=float(prob["uniform_bound"]) if prob["prior"] == "uniform" else None)
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
    n = Y.shape[0]
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
    out = dict(err_t = np.linalg.norm(med - truth, axis=0) / np.linalg.norm(truth, axis=0))
    for name, m in [("train", t <= T_train), ("test", t > T_train)]:
        if m.any():
            out[name] = dict(
                rel_err=np.linalg.norm(med[:, m] - truth[:, m]) / np.linalg.norm(truth[:, m]),
                cov=np.mean((truth[:, m] >= lo[:, m]) & (truth[:, m] <= hi[:, m])),
                width=np.mean(hi[:, m] - lo[:, m]),
                crps=crps[:, m].mean())
    return out


def main(noise=0.005, T_train=6.0, T_test=12.0, K=None, n_pf=500, system="original", deriv="dbr",
         lam=None, k_dbr=None, dbr_edge="guo", prior="tikhonov", uniform_bound=10.0):
    """noise   : relative snapshot noise, std of variable j = noise x its clean training range (0.05 = 5%)
       T_train : fit on [0, T_train]
       T_test  : extrapolate and score on (T_train, T_test]
       K       : training snapshots per trajectory; None keeps the default sampling rate
                 (300 snapshots over 6 time units) whatever T_train is
       n_pf    : number of posterior draws pushed forward through the ROM (<= 4 x 2000)
       system  : name of a system in toy_systems.SYSTEMS
       deriv   : "dbr" (De Brabanter, Guo et al.), "lpr" (Savitzky-Golay) or "fd" (finite differences)
       lam     : None -> per-row fixed-point lam (Guo et al.); a number -> fixed lam for all rows
                 (their convention: penalty lam^2; the old fixed Gamma = 0.1 I is lam = 0.316)
       k_dbr   : number of De Brabanter terms (None -> K // 30)
       dbr_edge: "guo" (keep all snapshots, as their code) or "trim" (drop the k boundary snapshots)
       prior   : "tikhonov" (Gamma_i = lam_i^2 I), "wide" (noninformative Gaussian, Gamma = 1e-8 I) or
                 "flat" (noninformative improper uniform, Gamma = 0) or "uniform" (proper U(-b, b));
                 wide/flat/uniform ignore lam
       uniform_bound : b of the uniform prior (prior = "uniform" only)"""
    K_given = K is not None
    if K is None:
        K = int(round(T_train * 299 / 6.0)) + 1
    # figures go to pdfs/<this file>_pdfs/, tagged with the run settings
    here, stem = os.path.split(os.path.splitext(os.path.abspath(__file__))[0])
    out_dir = os.path.join(here, "pdfs", stem + "_pdfs")
    os.makedirs(out_dir, exist_ok=True)
    
    tag = (("" if system == "original" else f"{system}_") +
           f"noise{noise:g}_Ttrain{T_train:g}_Ttest{T_test:g}" + (f"_K{K}" if K_given else "") +
           f"_n_pf{n_pf}_{deriv}" + (f"_k{k_dbr}" if k_dbr is not None else "") +
           ("_trim" if deriv == "dbr" and dbr_edge == "trim" else "") +
           ("" if lam is None or prior != "tikhonov" else f"_lam{lam:g}") +
           ("" if prior == "tikhonov" else f"_{prior}") +
           (f"{uniform_bound:g}" if prior == "uniform" else ""))
    out = lambda name: os.path.join(out_dir, f"{name}_{tag}.pdf")
    print(f"system = {system}, noise = {100 * noise:g}% of range, train on [0, {T_train}] with K = {K}, "
          f"test on ({T_train}, {T_test}], prior = {prior}")
    
    prob = build_problem(noise=noise, T_train=T_train, K=K, T_test=T_test, system=system,
                         deriv=deriv, lam=lam, k_dbr=k_dbr, dbr_edge=dbr_edge, prior=prior,
                         uniform_bound=uniform_bound)
    r_, d = prob["mu"].shape

    names = ["c"] + [f"A{k}" for k in range(r)] + [f"H{k}" for k in range(d - 1 - r)]
    results = []
    for i in range(r_):
        s, diag = sample_row(prob, i, return_diagnostics=True)  # chains x draws x d (MCMC samples for row i)
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
        ws = diag["warmup_fields"]["num_steps"]          # chains x warmup iterations
        print(f"  warmup cost: {ws.mean():.1f} leapfrog steps/iteration on average "
              f"({ws[:, :150].mean():.1f} in the first 150 iterations), "
              f"{int(ws.sum())} gradient evaluations over all chains")
        
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
              for lab, Y in [("analytical", Y_exact), ("MCMC", Y_mcmc)]}
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
        for Y, lab, col in [(Y_exact, "analytical Gaussian", "C0"), (Y_mcmc, "MCMC", "C1")]:
            lo, med, hi = np.percentile(Y[:, k], [2.5, 50, 97.5], axis=0)
            ax.fill_between(t, lo, hi, color=col, alpha=0.25, label=f"{lab} 95%")
            ax.plot(t, med, color=col, lw=1)
        ax.plot(t, prob["Q_true"][k], "k--", lw=0.8, label="truth")
        ax.plot(prob["t"], prob["Q"][k], "k.", ms=1, label="training observation")
        ax.plot(t[t > T_train], prob["Q_test"][k], ".", color="0.6", ms=1, label="test observations")
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
    ap.add_argument("--noise", type=float, default=0.005, help="relative snapshot noise: std of each variable = noise x its clean training range "
                         "(0.05 = 5%%)")
    ap.add_argument("--T_train", type=float, default=6.0, help="training window [0, T_train]")
    ap.add_argument("--T_test", type=float, default=12.0, help="test window (T_train, T_test]")
    ap.add_argument("--K", type=int, default=None,
                    help="training snapshots per trajectory (default: keep dt = 6/299)")
    ap.add_argument("--n_pf", type=int, default=500,
                    help="posterior draws pushed forward (at most 8000 = 4 chains x 2000)")
    ap.add_argument("--system", default="original", choices=list(SYSTEMS),
                    help="true ROM generating the data (defined in toy_systems.py)")
    ap.add_argument("--deriv", default="dbr", choices=["dbr", "lpr", "fd"],
                    help="derivative estimates: De Brabanter (Guo et al.), Savitzky-Golay, or finite differences")
    ap.add_argument("--k_dbr", type=int, default=None,
                    help="number of De Brabanter terms (default K // 30)")
    ap.add_argument("--dbr_edge", default="guo", choices=["guo", "trim"],
                    help="De Brabanter boundary: keep all snapshots as their code, or drop the k edge snapshots")
    ap.add_argument("--lam", type=float, default=None,
                    help="fixed lam for all rows, penalty lam^2 (default: per-row fixed point from lam = 50)")
    ap.add_argument("--prior", default="tikhonov", choices=["tikhonov", "wide", "flat", "uniform"],
                    help="tikhonov: Gamma_i = lam_i^2 I (fixed point or --lam); wide: noninformative Gaussian "
                         "(Gamma = 1e-8 I); flat: noninformative improper uniform (Gamma = 0); "
                         "uniform: proper U(-b, b) on every entry, b = --uniform_bound")
    ap.add_argument("--uniform_bound", type=float, default=10.0,
                    help="half-width b of the uniform prior U(-b, b) (prior = uniform only)")
    args = ap.parse_args()
    main(noise=args.noise, T_train=args.T_train, T_test=args.T_test, K=args.K, n_pf=args.n_pf,
         system=args.system, deriv=args.deriv, lam=args.lam, k_dbr=args.k_dbr, dbr_edge=args.dbr_edge,
         prior=args.prior, uniform_bound=args.uniform_bound)