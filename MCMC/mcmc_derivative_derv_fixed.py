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
    (default: De Brabanter weighted difference quotients, their ddt_localpoly).
    Two deviations from their code by default, needed when the noise is small (see below):
      * the first and last k snapshots of every trajectory are dropped (--dbr_edge trim). Their W leaves
        the first and last derivative at ZERO, which are O(1)-wrong targets at the initial conditions;
      * k = min(plug-in bias-variance choice, K // 30) (--k_dbr auto) instead of always K // 30. The fixed
        k = 10 has an O(dt^2 k^2 q''') bias that the plug-in sigma^2 then treats as noise.
    At noise 0 (toy, 20 x 300 snapshots, closed form = NUTS) these cut the operator error from 8.7% to
    0.36% and the training-window rollout error of the posterior mean from 2.0% to 0.16%. With noise >= 1%
    k stays at K // 30 and trimming also lowers the operator error (e.g. 1%: 10.7% -> 4.2%).
    --dbr_edge guo --k_dbr K30 reproduces their code exactly.
  - Gamma_i = lam_i^2 I per row, lam_i from their Bayesian fixed-point iteration (Algorithm 2,
    case 2 of bayes.construct_posterior) started at lam = 50, tol 1e-3, at most 50 iterations.
  - sigma_i^2 = (||D mu_i - r_i||^2 + lam_i^2 ||mu_i||^2) / K, as in their code.
Once lam_i and sigma_i^2 are fixed, the target is exactly Gaussian.
NUTS must reproduce (mu_i, Sigma_i) up to Monte Carlo error. If it does not, the bug is in the
log-posterior or the sampler, not in the method.

Need to swap in our own D, R, Gamma, sigma2 in `build_problem` to run this on real data.

Prior (--prior) of MCMC ONLY. The analytical posterior ALWAYS uses the Gaussian Tikhonov prior above
(lam_i from the fixed point, or the fixed --lam), and sigma_i^2 always comes from that fit; --prior only changes
the prior NUTS samples with (same likelihood, same sigma_i^2), so MCMC and the analytical posterior differ in
the prior alone. With --prior tikhonov they are the same target, which is the sampler check above.
  tikhonov  (default) o_i ~ N(0, sigma_i^2 Gamma_i^{-1}), Gamma_i = lam_i^2 I: the analytical posterior's prior.
  wide           Gaussian with Gamma = 1e-8 I: a proper prior with sd = 1e4 * sigma_i per entry (noninformative).
  flat           improper flat prior (no prior term): the likelihood alone (OLS posterior, needs D full rank).
  uniform        o_ij ~ U(-b, b) independently, b = --uniform_bound (default 10): the likelihood truncated to
                 the box [-b, b]^d (printed: how close the box comes to the posterior).
  laplace        o_ij ~ Laplace(0, b)          independently, b = --prior_scale (default 1): sparsity-promoting
  cauchy         o_ij ~ Cauchy(0, b)           independently: heavy tails, weak shrinkage of large entries
  student_t      o_ij ~ Student-t(nu, 0, b)    independently, nu = --prior_df (default 3; nu = 1 is Cauchy)
  horseshoe      regularized horseshoe (Carvalho et al. 2010; Piironen & Vehtari 2017):
                     o_ij = z_ij * tau_i * lt_ij,  z_ij ~ N(0, 1),  lam_ij ~ C+(0, 1),  tau_i ~ C+(0, tau0_i),
                     lt_ij^2 = c_i^2 lam_ij^2 / (c_i^2 + tau_i^2 lam_ij^2),  c_i^2 ~ Inv-Gamma(2, 2 s^2)
                 Local scales lam let single entries escape shrinkage, the global tau (learned) sets how much
                 the rest are shrunk, and the slab c caps the size of the unshrunk entries (s = --slab_scale,
                 default 2). tau0_i = p0 / (d - p0) * sigma_i / sqrt(K), p0 = --hs_p0 = guessed number of
                 nonzero entries per row (default 3). Sampled collapsed: the likelihood is Gaussian in o, so o
                 is integrated out given the scales (d x d algebra), NUTS samples only (tau, lam, c), and o is
                 drawn exactly from N(o | scales, r_i). Exact, and free of the o-vs-scale funnel.
  For every prior but tikhonov and horseshoe the run prints the "prior pull": the first-order shift
  Sigma_i (grad log p(mu_i) + Gamma_i mu_i / sigma_i^2) of the posterior mean from swapping the Gaussian prior
  for p, in posterior sd. << 1 -> MCMC ~ analytical; >~ 0.3 -> visibly different by design.

Data-generating system: --system damped | weak_damping | limit_cycle (defined in toy_systems.py).

Usage:
    python mcmc_derivative.py                                    # dbr (trimmed, auto k) + lam fixed point
    python mcmc_derivative.py --noise 0                          # noise-free check
    python mcmc_derivative.py --dbr_edge guo --k_dbr K30         # exactly Guo et al.'s ddt_localpoly
    python mcmc_derivative.py --noise 0.05 --T_train 6 --T_test 12 --K 300 --n_pf 100 --system limit_cycle
    python mcmc_derivative.py --deriv sg --lam 0.316             # old setup: SG derivatives, Gamma = 0.1 I
    python mcmc_derivative.py --prior flat                       # MCMC: noninformative (flat) prior
    python mcmc_derivative.py --prior wide                       # MCMC: noninformative, proper Gaussian
    python mcmc_derivative.py --prior uniform --uniform_bound 10   # MCMC: proper uniform U(-10, 10) per entry
    python mcmc_derivative.py --prior laplace --prior_scale 0.1    # MCMC: Laplace(0, 0.1) per entry
    python mcmc_derivative.py --prior student_t --prior_df 3 --prior_scale 0.1
    python mcmc_derivative.py --prior horseshoe --hs_p0 3 --slab_scale 2
    python mcmc_derivative.py --sweep                            # noise sweep 0, 1, ..., 10, 15, 20%
    python mcmc_derivative.py --sweep 0 0.05 0.1 --n_pf 200      # custom noise levels
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


# 1. Toy problem: the data-generating quadratic ROM  dq/dt = c + A q + H (q kron q)
#    The systems (damped, weak_damping, limit_cycle) live in toy_systems.py
from toy_systems import r, compact_kron, true_operators, rom_rhs, operators_to_matrix, SYSTEMS, system_tag

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


def debrabanter_auto_k(Q, dt, k_max=None):
    """Plug-in choice of the number of De Brabanter terms k (bias-variance tradeoff).

    For the weights w_j = 6 j^2 / (k (k+1) (2k+1)) the interior estimator has
        bias     = dt^2 q''' (3k^2 + 3k - 1) / 30
        variance = 3 s^2 / (dt^2 k (k+1) (2k+1))          (s = snapshot noise std)
    k minimizes the mean squared error summed over the states, with
        s^2         from second differences: mean((q_{i+1} - 2 q_i + q_{i-1})^2) / 6
        mean q'''^2 from a wide (~K/3 points) quintic Savitzky-Golay pilot, minus the pilot's own
                    noise variance s^2 sum(c^2) (a narrow pilot is noise-dominated: q''' amplifies
                    noise by ~1/dt^3, which would make k independent of the noise level).
    Without noise this gives k ~ 3 here; with noise k grows like s^(2/7).
    Guo et al. use the fixed k = K // 30, which is tuned for noisy data and leaves an O(dt^2 k^2 q''')
    bias that dominates when the noise is small.
    k_max: the pointwise-optimal k overweights variance for OpInf, because the regression averages
    the noise in R over all snapshots but not its bias; debrabanter_derivative therefore caps k at
    K // 30 (shrink Guo's window when the bias dominates, never widen it).
    """
    K = Q.shape[1]
    k_max = k_max or max(1, K // 4)
    s2 = np.mean(np.diff(Q, n=2, axis=1)**2, axis=1) / 6.0                # r,
    w_pilot = max(11, 2 * (K // 6) + 1)                                   # odd, ~K/3
    w_pilot = min(w_pilot, K if K % 2 else K - 1)
    q3 = savgol_filter(Q, w_pilot, 5, deriv=3, delta=dt, axis=1, mode="interp")
    c3 = savgol_coeffs(w_pilot, 5, deriv=3, delta=dt)
    B = np.mean(q3**2, axis=1) - s2 * np.sum(c3**2)                       # r,
    B = np.maximum(B, 1e-3 * np.mean(q3**2, axis=1) + 1e-300)
    ks = np.arange(1, k_max + 1)
    bias2 = (dt**2 * (3 * ks**2 + 3 * ks - 1) / 30.0)**2
    var = 3.0 / (dt**2 * ks * (ks + 1) * (2 * ks + 1))
    mse = bias2[:, None] * B[None, :] + var[:, None] * s2[None, :]         # k x r
    return int(ks[np.argmin(mse.sum(axis=1))])


def debrabanter_derivative(Q, dt, k=None, edge="onesided"):
    """First derivative by De Brabanter et al. (JMLR 2013) weighted symmetric difference
    quotients, following Guo et al.'s `ddt_localpoly` (euler1D.py, cmame2022):

        interior  (k <= i < K-k): dq_i = sum_{j=1}^{k} w_j (q_{i+j} - q_{i-j}) / (2 j dt),
                                  w_j = 6 j^2 / (k (k+1) (2k+1))
        boundary  (1 <= i < k and mirrored): the first i weights, renormalized to sum to 1
        end rows  i = 0 and i = K-1:
            edge = "guo"      -> left at ZERO, exactly as their matrix W. These are wrong targets
                                 (true q' is O(1) at the initial condition) that the Gaussian
                                 likelihood treats as data; at low noise they dominate the fit.
            edge = "onesided" -> second-order one-sided differences (default):
                                 dq_0 = (-3 q_0 + 4 q_1 - q_2) / (2 dt), mirrored at i = K-1

    With noise the end rows and the reduced boundary stencils (1 <= i < k) are also high-variance
    estimates, so build_problem's default drops all of them (dbr_edge = "trim").

    Q : r x K noisy snapshots
    k : number of terms; None -> K // 30 (their code), "auto" -> min(debrabanter_auto_k, K // 30)
    Returns dQ (r x K) and k. The states themselves are not smoothed.
    """
    K = Q.shape[1]
    if k is None:
        k = K // 30
    elif k == "auto":
        k = debrabanter_auto_k(Q, dt, k_max=max(1, K // 30))
    k = int(k)
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
    if edge == "onesided":
        W[0, :3] = np.array([-3.0, 4.0, -1.0]) / (2 * dt)
        W[K - 1, -3:] = np.array([1.0, -4.0, 3.0]) / (2 * dt)
    elif edge != "guo":
        raise ValueError(f"unknown edge = {edge!r}")
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

ENTRY_PRIORS = ("laplace", "cauchy", "student_t")   # independent per-entry priors, location 0, scale b
PRIORS = ("tikhonov", "wide", "flat", "uniform") + ENTRY_PRIORS + ("horseshoe",)   # MCMC priors (--prior)
WIDE_GAMMA = 1e-8                                    # prior precision / sigma2 of "wide": prior sd = 1e4 sigma_i


def entry_prior(kind, scale, df):
    """Prior of every operator entry: Laplace(0, b), Cauchy(0, b) or Student-t(df, 0, b), b = scale."""
    if kind == "laplace":
        return dist.Laplace(0.0, scale)
    if kind == "cauchy":
        return dist.Cauchy(0.0, scale)
    if kind == "student_t":
        return dist.StudentT(df, 0.0, scale)
    raise ValueError(f"unknown entry prior {kind!r}")


def entry_prior_label(kind, scale, df):
    return {"laplace": f"Laplace(0, {scale:g})", "cauchy": f"Cauchy(0, {scale:g})",
            "student_t": f"Student-t({df:g}, 0, {scale:g})"}[kind]


def build_problem(noise=0.005, T_train=6.0, K=300, n_traj=20,
                  deriv="dbr", window=None, poly=3, k_dbr="auto", dbr_edge="trim", lam=None, lam0=50.0,
                  tol=1e-3, T_test=None, system="damped", prior="tikhonov", uniform_bound=10.0,
                  prior_scale=1.0, prior_df=3.0, hs_p0=3.0, slab_scale=2.0, noise_ref="pooled",
                  clean_ic=False):
    """Guo et al. framework: D is always built from the RAW noisy states; only the derivative
    target R is estimated. Row i has the Tikhonov penalty lam_i^2 ||o_i||^2, i.e. the prior
    precision Gamma_i = lam_i^2 I (their convention: lam is the square root of the penalty weight).
    The analytical posterior (mu, Sigma) and sigma2 always come from this Tikhonov fit, whatever the prior.

       deriv = "dbr" : De Brabanter weighted difference quotients (their ddt_localpoly; default)
       deriv = "sg" : Savitzky-Golay local-polynomial derivatives (GCV window)
       deriv = "fd"  : second-order finite differences
       k_dbr : number of De Brabanter terms. "auto" (default) -> min(plug-in bias-variance choice, K // 30):
               Guo's K // 30 when the data are noisy, smaller when the noise estimate is small;
               None -> K // 30 (their code); an int -> that k
       dbr_edge : "trim"     -> drop the first and last k snapshots of every trajectory from D and R (default)
                  "guo"      -> keep every snapshot, exactly as their code (reduced stencils in the first
                                and last k points; the very first and last derivative are ZERO)
                  "onesided" -> as "guo" but the first/last derivative by one-sided differences
                                (fine without noise, high-variance outliers with noise)
       prior : the prior of MCMC only (see PRIORS and the module docstring): "tikhonov" (default, the analytical
               posterior's own Gaussian prior), "wide", "flat", "uniform" (U(-uniform_bound, uniform_bound)),
               "laplace" / "cauchy" / "student_t" (scale prior_scale, Student-t dof prior_df) or "horseshoe"
               (hs_p0 = guessed nonzeros per row, slab_scale = s). Never changes mu, Sigma or sigma2.
       lam   : None -> per-row lam_i by their fixed-point iteration from lam0 = 50,
               tol = 1e-3, at most 50 iterations (Algorithm 2 / case 2);
               a number -> the same fixed lam for every row (penalty lam^2)
       noise         : relative snapshot noise: std of variable j = noise * range (max - min) of
                       variable j over the clean training snapshots (0.05 = 5%), where the range is
       noise_ref     : "pooled"     -> over all trajectories together: one std per variable, the same for
                                       every trajectory (default)
                       "trajectory" -> over each trajectory separately: every trajectory gets noise x its
                                       own range (a trajectory with a small range gets a small std)
       clean_ic      : True -> the first snapshot of every trajectory is noise-free (the initial condition is
                       known exactly); every other snapshot is noisy as usual, with the same noise draws
       T_test        : if > T_train, also simulate trajectory 0 on (T_train, T_test] with the same dt as
                       held-out test data for time extrapolation. Only [0, T_train] enters D and R,
                       and the training data are identical to T_test=None."""
    if prior not in PRIORS:                      # the prior only enters MCMC (row_model / sample_row)
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
    Qs_clean, Z_train = [], []
    for l in range(n_traj):                      # several ICs -> better identifiability
        
        # random initial condition in [-1.5, 1.5]^r
        q0_rand = rng.uniform(-1.5, 1.5, r)
        Q_clean = solve_ivp(rom_rhs(O_true), (0, t_full[-1]), q0_rand, t_eval=t_full,
                            rtol=1e-10, atol=1e-12).y
        Qs_clean.append(Q_clean)
        
        # Gaussian noise draws for all snapshots of this trajectory (same shape as Q_clean)
        Z_train.append(rng.standard_normal((r, K)))
        
        if l == 0:    # truth + held-out noisy data for trajectory 0
            q0_ref, Q_ref = q0_rand, Q_clean
            Z_test = test_rng.standard_normal((r, n_ext))
            
    # noise std per trajectory and variable (n_traj x r): noise x the range of the clean training data,
    # pooled over all trajectories (the same std for every trajectory) or of each trajectory on its own
    if noise_ref == "pooled":
        Q_clean_train = np.hstack([Qclean[:, :K] for Qclean in Qs_clean])
        noise_stds = np.tile(noise * np.ptp(Q_clean_train, axis=1), (n_traj, 1))
        print(f"noise = {100 * noise:g}% of each variable's range over all trajectories: "
              f"std = {np.array2string(noise_stds[0], precision=4)}")
    elif noise_ref == "trajectory":
        noise_stds = noise * np.stack([np.ptp(Qclean[:, :K], axis=1) for Qclean in Qs_clean])
        print(f"noise = {100 * noise:g}% of each variable's range in its own trajectory: std of trajectory 0 = "
              f"{np.array2string(noise_stds[0], precision=4)}, over all trajectories "
              f"{np.array2string(noise_stds.min(axis=0), precision=4)} to "
              f"{np.array2string(noise_stds.max(axis=0), precision=4)}")
    else:
        raise ValueError(f"unknown noise_ref = {noise_ref!r}")
    noise_std = noise_stds[0]                    # trajectory 0: the one predicted, scored and plotted
    
    Q_noisy_test = Q_ref[:, K:] + noise_std[:, None] * Z_test
    # held-out noisy test data of every trajectory, for plotting several trajectories; drawn after
    # trajectory 0's, so its test data (and all training data) are unchanged
    Q_tests = np.stack([Q_noisy_test] + [Qc[:, K:] + sd[:, None] * test_rng.standard_normal((r, n_ext))
                                         for Qc, sd in zip(Qs_clean[1:], noise_stds[1:])])   # n_traj x r x n_ext

    Qs_noisy, Ds, Rs, windows = [], [], [], []
    for Q_clean, Z, sd in zip(Qs_clean, Z_train, noise_stds):
        if clean_ic:                             # noise-free initial snapshot (same draws elsewhere)
            Z = Z.copy(); Z[:, 0] = 0.0
        Q = Q_clean[:, :K] + sd[:, None] * Z
        Qs_noisy.append(Q)

        keep = slice(0, K)                       # snapshots of this trajectory that enter D and R
        if deriv == "dbr":
            if dbr_edge not in ("onesided", "guo", "trim"):
                raise ValueError(f"unknown dbr_edge = {dbr_edge!r}")
            dQ, kk = debrabanter_derivative(Q, dt, k=k_dbr,
                                            edge="guo" if dbr_edge == "trim" else dbr_edge)
            windows.append(kk)
            if dbr_edge == "trim":
                keep = slice(kk, K - kk)
        elif deriv == "fd":
            dQ = np.gradient(Q, t, axis=1, edge_order=2)
        elif deriv == "sg":
            _, dQ, w = local_poly_smooth(Q, dt, window=window, poly=poly)   # smoothed states unused
            windows.append(w)
        else:
            raise ValueError(f"unknown deriv = {deriv!r}")
        Ds.append(Q[:, keep])
        Rs.append(dQ[:, keep])

    # D from the raw noisy states, R from the derivative estimates
    Q_all, R = np.hstack(Ds), np.hstack(Rs)                    # r x (sum of kept snapshots)
    K_all = Q_all.shape[1]

    # data matrix D = [1, q^T, (q kron q)^T]   (K x d), built from the raw noisy states
    D = np.hstack([np.ones((K_all, 1)), Q_all.T, compact_kron(Q_all).T])
    d = D.shape[1]

    # per-row regularization lam_i (penalty lam_i^2) and plug-in noise variance sigma2_i
    if lam is None:
        lams, sigma2, n_it, converged = guo_regularization(D, R, reg0=lam0, tol=tol)
        status = f"fixed point {'converged' if converged else 'NOT converged'} after {n_it} iterations"
    else:
        lams = np.full(r, float(lam))
        _, sigma2, _ = guo_posterior_params(D, R, lams)
        status = "fixed lam"

    # analytical posterior per row: mu_i = (G + lam_i^2 I)^{-1} D^T r_i, Sigma_i = sigma2_i (G + lam_i^2 I)^{-1}
    G = D.T @ D
    G = 0.5 * (G + G.T)
    Gamma = np.stack([l**2 * np.eye(d) for l in lams])           # r x d x d prior precision / sigma2
    P_inv = np.stack([np.linalg.inv(G + Gi) for Gi in Gamma])    # r x d x d
    mu = np.stack([P_inv[i] @ D.T @ R[i] for i in range(r)])     # r x d
    Sigma = sigma2[:, None, None] * P_inv                        # r x d x d

    unit = "terms" if deriv == "dbr" else "pts"
    if windows:
        wr = f"{min(windows)}" if min(windows) == max(windows) else f"{min(windows)}-{max(windows)}"
        info = f", {deriv} window = {wr} {unit}" + (f", edges {dbr_edge}" if deriv == "dbr" else "")
    else:
        info = ""
    print(f"[{deriv}] d = {d} params/row, K = {K_all} snapshots, cond(D^T D) = {np.linalg.cond(G):.2e}{info}")
    print(f"  regularization: {status}")
    for i in range(r):
        print(f"  row {i}: lam = {lams[i]:.4g} (penalty lam^2 = {lams[i]**2:.4g}), sigma2 = {sigma2[i]:.3e}, "
              f"cond(D^T D + lam^2 I) = {np.linalg.cond(G + Gamma[i]):.2e}")
    # horseshoe global-scale guess (Piironen & Vehtari 2017): tau0_i = p0 / (d - p0) * sigma_i / sqrt(K)
    tau0 = hs_p0 / (d - hs_p0) * np.sqrt(sigma2) / np.sqrt(K_all)
    if prior != "tikhonov":
        desc = {"wide": f"Gaussian, Gamma = {WIDE_GAMMA:g} I", "flat": "flat (improper)",
                "uniform": f"U(-{uniform_bound:g}, {uniform_bound:g}) on every entry",
                "horseshoe": f"regularized horseshoe, p0 = {hs_p0:g} of {d} entries per row, "
                             f"slab scale {slab_scale:g}, tau0 = {np.array2string(tau0, precision=3)}"}
        desc = desc.get(prior) or f"{entry_prior_label(prior, prior_scale, prior_df)} on every entry"
        print(f"  MCMC prior: {desc}; the analytical posterior keeps the Gaussian prior above")
    sd = np.sqrt(np.stack([np.diag(S) for S in Sigma]))
    if prior == "uniform":
        # distance from the analytical posterior mean to the nearest box face, in posterior sd (per row)
        margin = ((uniform_bound - np.abs(mu)) / sd).min(axis=1)
        print(f"  uniform box: nearest face at {np.array2string(margin, precision=1)} posterior sd per row "
              f"(> ~5: the box does not cut into the posterior)")
        if np.any(margin < 5):
            print("  WARNING: the uniform box truncates the posterior")
    if prior not in ("tikhonov", "horseshoe"):
        # MCMC target = analytical posterior N(mu_i, Sigma_i) / Gaussian prior x p(o)
        #             = N(mu_i, Sigma_i) exp(o^T Gamma_i o / (2 sigma2_i)) p(o),
        # whose mode is ~ mu_i + Sigma_i (grad log p(mu_i) + Gamma_i mu_i / sigma2_i) to first order.
        # Shift in posterior sd, max over the entries of each row. (flat / uniform inside the box: grad = 0.)
        if prior in ENTRY_PRIORS:
            p = entry_prior(prior, prior_scale, prior_df)
            grad = np.asarray(jax.vmap(jax.grad(lambda x: p.log_prob(x).sum()))(jnp.asarray(mu)))   # r x d
        elif prior == "wide":
            grad = -WIDE_GAMMA * mu / sigma2[:, None]
        else:
            grad = np.zeros_like(mu)
        grad = grad + np.einsum("ijk,ik->ij", Gamma, mu) / sigma2[:, None]     # undo the Gaussian prior
        pull = (np.abs(np.einsum("ijk,ik->ij", Sigma, grad)) / sd).max(axis=1)
        print(f"  prior pull (MCMC vs. analytical mean): {np.array2string(pull, precision=3)} posterior sd per row "
              f"(<< 1: MCMC ~ analytical; >~ 0.3: departs from it by design)")
    if lam is None and not converged:
        print("  WARNING: the lam fixed point did not converge; the posterior uses the last iterate")
    if n_ext:
        print(f"test window: ({T_train:.1f}, {t_full[-1]:.1f}], {n_ext} held-out snapshots of trajectory 0")
    # ROM initial condition = first raw noisy snapshot (as in their plot_euler.py: init_ = Vr^T noisy[:, 0])
    return dict(t=t, q0=Qs_noisy[0][:, 0], Q=Qs_noisy[0],
                Qs=np.stack(Qs_noisy),                               # n_traj x r x K noisy snapshots
                D=D, R=R, Gamma=Gamma, lam=lams, sigma2=sigma2, mu=mu, Sigma=Sigma,
                O_true=O_true, deriv=deriv, windows=windows, prior=prior, uniform_bound=uniform_bound,
                prior_scale=prior_scale, prior_df=prior_df, lam_input=lam,
                hs_p0=hs_p0, slab_scale=slab_scale, tau0=tau0,
                q0_ref=q0_ref, t_full=t_full, T_train=T_train, Q_ref=Q_ref, Q_test=Q_noisy_test,
                noise_std=noise_std, noise_stds=noise_stds, noise_ref=noise_ref,
                Qs_clean_full=np.stack(Qs_clean),                   # n_traj x r x len(t_full), clean
                Q_tests=Q_tests)                                     # n_traj x r x n_ext, noisy held-out



# 2. NumPyro model for one row. Written as prior + likelihood (not as the
#    closed-form Gaussian) to check the log-posterior.
def row_model(D, r_i, Gamma, sigma2_i, flat=False, bound=None, entry=None, scale=1.0, df=3.0,
              tau0=None, slab_scale=2.0):
    # Bayesian model for one row: r_i | o ~ N(D o, sigma2_i I), o ~ N(0, sigma2_i Gamma^{-1})
    # flat=True: Gamma = 0 has no inverse, so the prior is an improper uniform density on R^d
    # (no prior term in the log-density); the likelihood alone defines the posterior.
    # bound=b: proper uniform prior o_j ~ U(-b, b) independently. Sampled as u ~ U(-1, 1)^d, o = b u
    # (constant Jacobian, same posterior).
    # The fixed U(-1, 1) keeps the box transform free of traced arguments; U(-b, b)
    # with a traced b breaks NumPyro's separate warmup()/run() calls
    # NUTS moves in logit(u) space, so the box constraint is handled exactly.
    # entry="laplace" / "cauchy" / "student_t": every entry independently ~ entry_prior(entry, scale, df).
    # tau0=t: regularized horseshoe with global scale tau ~ C+(0, t) and slab scale slab_scale, collapsed:
    # given the prior sds v = tau lt, o ~ N(0, diag(v^2)) and r_i ~ N(D o, sigma2_i I) are jointly Gaussian, so
    #   log p(r_i | v) = -1/2 log det B + 1/2 u^T B^{-1} u + const,   B = I + V G V,  u = V D^T r_i / sigma2_i,
    #   o | v, r_i ~ N(V B^{-1} u, V B^{-1} V),                       V = diag(v),  G = D^T D / sigma2_i.
    # NUTS samples only the scales; o = V L^{-T} (L^{-1} u + z), B = L L^T, z ~ N(0, I) is an exact draw.
    d = D.shape[1]

    if bound is not None:
        u = numpyro.sample("u", dist.Uniform(-jnp.ones(d), jnp.ones(d)).to_event(1))
        o = numpyro.deterministic("o", bound * u)
    elif entry is not None:
        o = numpyro.sample("o", entry_prior(entry, scale, df).expand([d]).to_event(1))
    elif tau0 is not None:
        tau = numpyro.sample("tau", dist.HalfCauchy(tau0))                          # global scale
        lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones(d)).to_event(1))      # local scales
        c2 = numpyro.sample("c2", dist.InverseGamma(2.0, 2.0 * slab_scale**2))     # slab width^2 (nu = 4)
        lt = jnp.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))                         # regularized local scales
        v = tau * lt                                                                 # prior sd of each entry
        B = jnp.eye(d) + v[:, None] * (D.T @ D / sigma2_i) * v[None, :]
        L = jnp.linalg.cholesky(B)
        w = jax.scipy.linalg.solve_triangular(L, v * (D.T @ r_i) / sigma2_i, lower=True)   # L^{-1} u
        numpyro.factor("marginal_likelihood", 0.5 * w @ w - jnp.sum(jnp.log(jnp.diag(L))))
        z = numpyro.sample("z", dist.Normal(jnp.zeros(d), 1.0).to_event(1))
        numpyro.deterministic("o", v * jax.scipy.linalg.solve_triangular(L.T, w + z, lower=False))
        return                                                    # o is integrated out: no likelihood site
    elif flat:
        o = numpyro.sample("o", dist.ImproperUniform(dist.constraints.real_vector, (), (d,)))
    else:   # Gaussian prior: o_i ~ N(0, sigma2_i * inv(Gamma))
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
    # The horseshoe posterior has strongly varying curvature (entries near zero vs. not), so it uses a
    # higher target acceptance (smaller steps) to avoid divergences.
    kernel = NUTS(row_model, dense_mass=True, target_accept_prob=0.95 if prob["prior"] == "horseshoe" else 0.9)

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)

    # data for this row: data matrix D, observed data r_i, prior precision Gamma_i = lam_i^2 I, noise variance
    # sigma2_i (always from the Tikhonov fit), and the MCMC prior selected by prob["prior"]
    data = dict(D=jnp.asarray(prob["D"]), r_i=jnp.asarray(prob["R"][i]),
                Gamma=jnp.asarray(WIDE_GAMMA * np.eye(prob["D"].shape[1]) if prob["prior"] == "wide"
                                  else prob["Gamma"][i]),     # Gaussian priors: tikhonov's own or "wide"
                sigma2_i=float(prob["sigma2"][i]),
                # sigma2_i=float(0.001),
                flat=bool(prob["prior"] == "flat"),
                bound=float(prob["uniform_bound"]) if prob["prior"] == "uniform" else None,
                entry=prob["prior"] if prob["prior"] in ENTRY_PRIORS else None,
                scale=float(prob["prior_scale"]), df=float(prob["prior_df"]),
                tau0=float(prob["tau0"][i]) if prob["prior"] == "horseshoe" else None,
                slab_scale=float(prob["slab_scale"]))
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
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))

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
        
        # ax = axes[row, 2]
        # ranks = stats.rankdata(samples[:, :, j].ravel()).reshape(n_chain, n_draw)
        # bins = np.linspace(0, n_chain * n_draw, 21)
        # for c in range(n_chain):
        #     h, _ = np.histogram(ranks[c], bins=bins)
        #     ax.step(bins[:-1], h, where="post", color=CHAIN_COLORS[c], lw=1.2)
        # exp = n_draw / 20
        # lo_b, hi_b = stats.binom.ppf([0.005, 0.995], n_draw, 1 / 20)   # 99% band for one bin
        # ax.axhspan(lo_b, hi_b, color="0.88", zorder=0)
        # ax.axhline(exp, color="#1f1f1e", lw=0.8, ls="--")
        # ax.set_ylim(0, 2.2 * exp)
        # ax.set_title(f"{names[j]}: rank plot (grey: 99% band if uniform)", fontsize=9)
        # ax.set_xlabel("rank among all draws", fontsize=8)
        # (d) autocorrelation per chain
        
        ax = axes[row, 2]
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
    
    # # (g) per-chain means vs. the exact mean, in units of the exact sd (all d entries)
    # ax = axes[3, 2]
    # z = (samples.mean(1) - mu_i) / np.sqrt(np.diag(Sigma_i))           # chains x d
    # for c in range(n_chain):
    #     ax.plot(np.arange(d) + (c - 1.5) * 0.12, z[c], "o", ms=4, color=CHAIN_COLORS[c])
    # ax.axhline(0, color="#1f1f1e", lw=0.8)
    # ax.set_xticks(np.arange(d), names, fontsize=7)
    # ax.set_title("per-chain mean − exact mean (in exact sd)", fontsize=9)
    # # (h) most correlated pair: draws vs. the exact 95% ellipse
    
    ax = axes[3, 2]
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


def figure_dir():
    """Figures go to pdfs/<this file>_pdfs/."""
    here, stem = os.path.split(os.path.splitext(os.path.abspath(__file__))[0])
    out_dir = os.path.join(here, "pdfs", stem + "_pdfs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_tag(noise_label, T_train, T_test, K, n_pf, system, deriv, lam, k_dbr, dbr_edge, prior, uniform_bound,
            prior_scale=1.0, prior_df=3.0, hs_p0=3.0, slab_scale=2.0, noise_ref="pooled", n_traj=20,
            clean_ic=False):
    """File-name tag with the run settings; noise_label is e.g. "noise0.05" or "noisesweep"."""
    return (("" if system == "damped" else f"{system_tag(system)}_") +
            f"{noise_label}" + ("" if noise_ref == "pooled" else "_noiseref-traj") +
            ("" if n_traj == 20 else f"_ntraj{n_traj}") +
            ("_cleanIC" if clean_ic else "") +
            f"_Ttrain{T_train:g}_Ttest{T_test:g}" + (f"_K{K}" if K is not None else "") +
            f"_n_pf{n_pf}_{deriv}" +
            ("" if deriv != "dbr" or k_dbr == "auto" else f"_k{k_dbr if k_dbr is not None else 'K30'}") +
            ("" if deriv != "dbr" or dbr_edge == "trim" else f"_{dbr_edge}edge") +
            ("" if lam is None else f"_lam{lam:g}") +
            ("" if prior == "tikhonov" else f"_{prior}") +
            (f"{uniform_bound:g}" if prior == "uniform" else "") +
            (f"_nu{prior_df:g}" if prior == "student_t" else "") +
            (f"_b{prior_scale:g}" if prior in ENTRY_PRIORS else "") +
            (f"_p0{hs_p0:g}_s{slab_scale:g}" if prior == "horseshoe" else ""))


def run_case(noise=0.005, T_train=6.0, T_test=12.0, K=None, n_pf=500, system="damped", deriv="dbr",
             lam=None, k_dbr="auto", dbr_edge="trim", prior="tikhonov", uniform_bound=10.0,
             prior_scale=1.0, prior_df=3.0, hs_p0=3.0, slab_scale=2.0, noise_ref="pooled", n_traj=20,
             clean_ic=False, diagnostics=True, out=None):
    """One noise level: build the problem, sample every row with NUTS, compare to the analytical
    posterior, and push n_pf draws of both posteriors through the ROM on [0, T_test].
    diagnostics=True : also keep the warmup, print the full per-row checks and save the mixing
                       figures to out(name) (used by main)
    diagnostics=False: plain NUTS and a one-line check per row (used by noise_sweep)
    The global rng is reset first, so every call uses the same ICs and unit noise draws (only the
    noise scale changes) and a sweep entry reproduces the single run at that noise."""
    global rng
    rng = np.random.default_rng(0)
    if K is None:
        K = int(round(T_train * 299 / 6.0)) + 1
    print(f"system = {system}, noise = {100 * noise:g}% of range, train on [0, {T_train}] with "
          f"{n_traj} trajectories x K = {K} snapshots, "
          f"test on ({T_train}, {T_test}], MCMC prior = {prior} (analytical: tikhonov)")

    prob = build_problem(noise=noise, T_train=T_train, K=K, n_traj=n_traj, T_test=T_test, system=system,
                         deriv=deriv, lam=lam, k_dbr=k_dbr, dbr_edge=dbr_edge, prior=prior,
                         uniform_bound=uniform_bound, prior_scale=prior_scale, prior_df=prior_df,
                         hs_p0=hs_p0, slab_scale=slab_scale, noise_ref=noise_ref, clean_ic=clean_ic)
    r_, d = prob["mu"].shape

    names = ["c"] + [f"A{k}" for k in range(r)] + [f"H{k}" for k in range(d - 1 - r)]
    results = []
    for i in range(r_):
        if not diagnostics:
            s = sample_row(prob, i)                               # chains x draws x d
            res = compare_row(s, prob["mu"][i], prob["Sigma"][i])
            results.append(res)
            print(f"row {i}: min ESS = {res['ess'].min():.0f}, max R-hat = {res['rhat'].max():.3f}, "
                  f"KL(MCMC || analytic) = {res['kl']:.4f} (baseline ~ {res['kl_expected']:.4f})")
            continue
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
    # what the true operators generate from the noisy IC the push-forward starts from
    Q_noisy_ic = solve_ivp(rom_rhs(prob["O_true"]), (t[0], t[-1]), prob["q0"], t_eval=t,
                           rtol=1e-10, atol=1e-12).y
    print(f"\npush-forward: {len(Y_mcmc)}/{n_pf} MCMC and {len(Y_exact)}/{n_pf} analytical rollouts stayed finite")
    if len(Y_mcmc) and len(Y_exact):
        for k in range(r_):
            b_m = np.percentile(Y_mcmc[:, k], [2.5, 97.5], axis=0)
            b_e = np.percentile(Y_exact[:, k], [2.5, 97.5], axis=0)
            gap = np.abs(b_m - b_e).max() / (b_e[1] - b_e[0]).max()
            print(f"  q{k}: max band-edge gap / max band width = {gap:.3f}")

    # prediction skill vs. the noiseless truth, training window vs. extrapolation window
    scores = {lab: score_prediction(Y, t, prob["Q_ref"], T_train)
              for lab, Y in [("analytical", Y_exact), ("MCMC", Y_mcmc)] if len(Y)}
    hdr = (f"\n{'posterior':>10s} | {'window':>6s} | {'rel err':>7s} | {'95% cov':>7s} | "
           f"{'band width':>10s} | {'CRPS':>7s}")
    print(hdr); print("-" * len(hdr))
    for lab, sc in scores.items():
        for w in ["train", "test"]:
            if w in sc:
                v = sc[w]
                print(f"{lab:>10s} | {w:>6s} | {v['rel_err']:7.4f} | {v['cov']:7.2f} | "
                      f"{v['width']:10.4f} | {v['crps']:7.4f}")
    return dict(prob=prob, results=results, names=names, t=t, T_train=T_train, n_pf=n_pf,
                Y_mcmc=Y_mcmc, Y_exact=Y_exact, Q_noisy_ic=Q_noisy_ic, scores=scores)


def prior_labels(prob):
    """Legend names (analytical, MCMC) of the priors of a build_problem result."""
    prior, lam = prob["prior"], prob["lam_input"]
    gauss = "Gaussian prior, " + (r"$\lambda$ fixed point" if lam is None else rf"$\lambda$ = {lam:g}")
    analytical = gauss                        # always the Tikhonov posterior
    if prior == "wide":
        mcmc = r"Gaussian prior, $\Gamma = 10^{-8} I$"
    elif prior == "flat":
        mcmc = "flat prior"
    elif prior == "uniform":
        mcmc = f"U(-{prob['uniform_bound']:g}, {prob['uniform_bound']:g}) prior"
    elif prior in ENTRY_PRIORS:
        mcmc = f"{entry_prior_label(prior, prob['prior_scale'], prob['prior_df'])} prior"
    elif prior == "horseshoe":
        mcmc = f"horseshoe prior ($p_0$ = {prob['hs_p0']:g}, s = {prob['slab_scale']:g})"
    else:
        mcmc = analytical                     # tikhonov: same prior on both sides
    return {"analytical": analytical, "MCMC": mcmc}


def plot_bands(ax, case, k):
    """State k of one run_case result: 95% push-forward bands and medians (analytical vs. MCMC),
    truth, true operator from the noisy IC, and the training / test observations."""
    t, prob, T_train = case["t"], case["prob"], case["T_train"]
    pl = prior_labels(prob)
    for Y, lab, col in [(case["Y_exact"], f"analytical, {pl['analytical']}", "C0"),
                        (case["Y_mcmc"], f"MCMC, {pl['MCMC']}", "C1")]:
        if not len(Y):                        # every rollout blew up
            continue
        lo, med, hi = np.percentile(Y[:, k], [2.5, 50, 97.5], axis=0)
        ax.fill_between(t, lo, hi, color=col, alpha=0.25, label=f"{lab} 95%")
        ax.plot(t, med, color=col, lw=1)
    ax.plot(t, prob["Q_ref"][k], "k-", lw=1.0, label="truth (clean IC)")
    ax.plot(t, case["Q_noisy_ic"][k], "k--", lw=0.8, label="true operator from noisy IC")
    ax.plot(prob["t"], prob["Q"][k], "k.", ms=1, label="training observation")
    ax.plot(t[t > T_train], prob["Q_test"][k], ".", color="0.6", ms=1, label="test observations")
    ax.axvline(T_train, color="0.5", lw=0.8, ls=":")


def main(noise=0.01, T_train=6.0, T_test=12.0, K=None, n_pf=500, system="damped", deriv="dbr",
         lam=None, k_dbr="auto", dbr_edge="trim", prior="tikhonov", uniform_bound=10.0,
         prior_scale=1.0, prior_df=3.0, hs_p0=3.0, slab_scale=2.0, noise_ref="pooled", n_traj=20,
         clean_ic=False):
    """noise   : relative snapshot noise, std of variable j = noise x its clean training range (0.05 = 5%)
       T_train : fit on [0, T_train]
       T_test  : extrapolate and score on (T_train, T_test]
       K       : training snapshots per trajectory; None keeps the default sampling rate
                 (300 snapshots over 6 time units) whatever T_train is
       n_pf    : number of posterior draws pushed forward through the ROM (<= 4 x 2000)
       system  : name of a system in toy_systems.SYSTEMS
       deriv   : "dbr" (De Brabanter, Guo et al.), "sg" (Savitzky-Golay) or "fd" (finite differences)
       lam     : None -> per-row fixed-point lam (Guo et al.); a number -> fixed lam for all rows
                 (their convention: penalty lam^2; the old fixed Gamma = 0.1 I is lam = 0.316)
       k_dbr   : number of De Brabanter terms ("auto" -> min(plug-in, K // 30); None -> K // 30; or an int)
       dbr_edge: "trim" (drop the k boundary snapshots; default), "guo" (keep all snapshots with zero end
                 derivatives, as their code) or "onesided" (as "guo" with one-sided end derivatives)
       prior   : prior of MCMC only; the analytical posterior is always the tikhonov one (lam as above).
                 "tikhonov" (Gamma_i = lam_i^2 I), "wide" (Gaussian, Gamma = 1e-8 I), "flat" (improper),
                 "uniform" (U(-b, b)), "laplace" / "cauchy" / "student_t" (per entry, scale prior_scale) or
                 "horseshoe" (regularized, hs_p0 / slab_scale)
       uniform_bound : b of the uniform prior (prior = "uniform" only)
       prior_scale   : scale b of the laplace / cauchy / student_t prior
       prior_df      : degrees of freedom nu of the student_t prior
       hs_p0         : horseshoe: guessed number of nonzero entries per row (sets tau0)
       slab_scale    : horseshoe: slab scale s, the size up to which large entries stay unshrunk
       noise_ref     : range the noise refers to: "pooled" (all trajectories) or "trajectory" (each its own)
       clean_ic      : noise-free first snapshot of every trajectory (rollouts then start from the exact IC)
       n_traj        : number of training trajectories, each from its own random initial condition"""
    settings = dict(T_train=T_train, T_test=T_test, K=K, n_pf=n_pf, system=system, deriv=deriv, lam=lam,
                    k_dbr=k_dbr, dbr_edge=dbr_edge, prior=prior, uniform_bound=uniform_bound,
                    prior_scale=prior_scale, prior_df=prior_df, hs_p0=hs_p0, slab_scale=slab_scale,
                    noise_ref=noise_ref, n_traj=n_traj, clean_ic=clean_ic)
    # figures go to pdfs/<this file>_pdfs/, tagged with the run settings
    out_dir = figure_dir()
    tag = run_tag(f"noise{noise:g}", **settings)
    out = lambda name: os.path.join(out_dir, f"{name}_{tag}.pdf")

    case = run_case(noise=noise, diagnostics=True, out=out, **settings)
    prob, results, names, t, scores = case["prob"], case["results"], case["names"], case["t"], case["scores"]
    r_, d = prob["mu"].shape
    title = (f"noise level = {100 * noise:g}%"   # suptitle of every figure below
             + ("" if noise_ref == "pooled" else " of each trajectory's own range"))

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
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    # fig.savefig("qq_marginals.png", dpi=300)
    fig.savefig(out("qq_marginals"))

    # forest plot: posterior 95% interval of every operator entry, analytical vs. MCMC, against the truth
    # (as traj_vs_deriv_operators in mcmc_trajectory2.py)
    O_true, pl = prob["O_true"], prior_labels(prob)
    sd_an = np.sqrt(np.stack([np.diag(S) for S in prob["Sigma"]]))                     # r x d
    post = {"analytical": (prob["mu"], prob["mu"] - 1.96 * sd_an, prob["mu"] + 1.96 * sd_an),
            "MCMC": (np.stack([res["flat"].mean(0) for res in results]),
                     *[np.stack([np.percentile(res["flat"], q, axis=0) for res in results]) for q in (2.5, 97.5)])}
    hdr = f"\n{'posterior':>10s} | {'op rel err':>10s} | {'op 95% cov':>10s} | {'op CI width':>11s}"
    print(hdr); print("-" * len(hdr))
    for lab, (m_, lo_, hi_) in post.items():
        print(f"{lab:>10s} | {np.linalg.norm(m_ - O_true) / np.linalg.norm(O_true):10.3f} | "
              f"{np.mean((O_true >= lo_) & (O_true <= hi_)):10.2f} | {np.mean(hi_ - lo_):11.3f}")
    fig, axes = plt.subplots(r_, 1, figsize=(11, 2.8 * r_), sharex=True)
    for i, ax in enumerate(axes):
        for k, (lab, (m_, lo_, hi_)) in enumerate(post.items()):
            x = np.arange(d) + (k - 0.5) * 0.25
            ax.errorbar(x, m_[i], yerr=[m_[i] - lo_[i], hi_[i] - m_[i]], fmt="o", ms=3, lw=1.4, color=f"C{k}",
                        label=f"{lab}, {pl[lab]}" if i == 0 else None)
        ax.plot(np.arange(d), O_true[i], "kx", ms=8, mew=2, label="truth" if i == 0 else None)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_ylabel(f"row {i}")
    axes[-1].set_xticks(np.arange(d), names)
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    fig.suptitle(title + " — operator posteriors (95% intervals)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out("operators"))

    fig, axes = plt.subplots(r_, 1, figsize=(8, 2.4 * r_), sharex=True)
    for k in range(r_):
        plot_bands(axes[k], case, k)
        axes[k].set_ylabel(f"$x_{k+1}$")
    axes[0].legend(fontsize=7, loc="upper right", ncol=3)
    axes[-1].set_xlabel("t")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    # fig.savefig("pushforward_bands.png", dpi=300)
    fig.savefig(out("pushforward_bands"))

    # relative error of the median prediction over time: growth past T_train
    fig, ax = plt.subplots(figsize=(7, 2.6))
    pl = prior_labels(prob)
    for (lab, sc), col in zip(scores.items(), ["C0", "C1"]):
        ax.semilogy(t, sc["err_t"], color=col, lw=1, label=f"{lab}, {pl[lab]}")
    ax.axvline(case["T_train"], color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("t"); ax.set_ylabel("rel. error of median")
    ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out("extrapolation_error"))
    print(f"saved {{qq_marginals, operators, pushforward_bands, extrapolation_error, mixing_row0-{r_ - 1}}}_{tag}.pdf "
          f"in {out_dir}")


# 6. Noise sweep: the same fit + push-forward at several noise levels
NOISE_SWEEP = [0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]


def noise_sweep(noises=NOISE_SWEEP, T_train=6.0, T_test=12.0, K=None, n_pf=500, system="damped",
                deriv="dbr", lam=None, k_dbr="auto", dbr_edge="trim", prior="tikhonov", uniform_bound=10.0,
                prior_scale=1.0, prior_df=3.0, hs_p0=3.0, slab_scale=2.0, noise_ref="pooled", n_traj=20,
                clean_ic=False):
    """Run run_case at every noise level (same ICs and unit noise draws, only scaled) and compare:
         pushforward_noise_sweep : push-forward bands, one row per state, one column per noise level;
                                   y-limits shared per row so band widths compare across noise levels
         noise_sweep_scores      : rel. error of the median, 95% coverage, band width and CRPS vs. noise,
                                   train (solid) and test (dashed), analytical vs. MCMC
    Other arguments as in main. The per-row mixing and QQ figures are skipped to keep the sweep cheap."""
    settings = dict(T_train=T_train, T_test=T_test, K=K, n_pf=n_pf, system=system, deriv=deriv, lam=lam,
                    k_dbr=k_dbr, dbr_edge=dbr_edge, prior=prior, uniform_bound=uniform_bound,
                    prior_scale=prior_scale, prior_df=prior_df, hs_p0=hs_p0, slab_scale=slab_scale,
                    noise_ref=noise_ref, n_traj=n_traj, clean_ic=clean_ic)
    out_dir = figure_dir()
    tag = run_tag("noisesweep", **settings)
    out = lambda name: os.path.join(out_dir, f"{name}_{tag}.pdf")

    cases = []
    for noise in noises:
        print(f"\n{'=' * 30} noise = {100 * noise:g}% {'=' * 30}")
        cases.append(run_case(noise=noise, diagnostics=False, **settings))

    # summary table over all noise levels
    hdr = (f"\n{'noise':>6s} | {'posterior':>10s} | {'finite':>7s} | {'window':>6s} | {'rel err':>7s} | "
           f"{'95% cov':>7s} | {'band width':>10s} | {'CRPS':>7s}")
    print(hdr); print("-" * len(hdr))
    for noise, case in zip(noises, cases):
        for lab, Y in [("analytical", case["Y_exact"]), ("MCMC", case["Y_mcmc"])]:
            for w in ["train", "test"]:
                v = case["scores"].get(lab, {}).get(w)
                if v is None:
                    continue
                print(f"{100 * noise:5g}% | {lab:>10s} | {len(Y):3d}/{n_pf:<3d} | {w:>6s} | {v['rel_err']:7.4f} | "
                      f"{v['cov']:7.2f} | {v['width']:10.4f} | {v['crps']:7.4f}")

    # push-forward bands: rows = states, columns = noise levels
    n = len(noises)
    fig, axes = plt.subplots(r, n, figsize=(3.2 * n, 2.4 * r + 0.6), sharex=True, sharey="row", squeeze=False)
    for col, (noise, case) in enumerate(zip(noises, cases)):
        for k in range(r):
            plot_bands(axes[k, col], case, k)
        axes[0, col].set_title(f"noise {100 * noise:g}%", fontsize=15)
        axes[-1, col].set_xlabel("t", fontsize=15)
        axes[-1, col].tick_params(labelsize=15)
    
    for k in range(r):
        # limits from the truth and the noisy data of every level; a band leaving them is clipped
        data = np.concatenate([np.concatenate([c["prob"]["Q_ref"][k], c["prob"]["Q"][k], c["prob"]["Q_test"][k]])
                               for c in cases])
        lo, hi = data.min(), data.max()
        axes[k, 0].set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        axes[k, 0].set_ylabel(f"$x_{k}$", fontsize=15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=15, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    fig.savefig(out("pushforward_noise_sweep"))

    # prediction skill vs. noise level
    x = 100 * np.asarray(noises)
    metrics = [("rel_err", "rel. error of median", True), ("cov", "95% band coverage", False),
               ("width", "mean 95% band width", True), ("crps", "CRPS", True)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.2))
    for ax, (key, title, log) in zip(axes, metrics):
        for lab, col in [("analytical", "C0"), ("MCMC", "C1")]:
            for w, ls in [("train", "-"), ("test", "--")]:
                y = [c["scores"].get(lab, {}).get(w, {}).get(key, np.nan) for c in cases]
                ax.plot(x, y, ls, color=col, marker="o", ms=3, lw=1,
                        label=f"{lab}, {prior_labels(cases[0]['prob'])[lab]}, {w}")
        if log:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("noise [% of range]", fontsize=15)
        ax.tick_params(labelsize=15)
    axes[1].axhline(0.95, color="0.5", lw=0.8, ls=":")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].tick_params(labelsize=15)
    axes[0].legend(fontsize=15)
    fig.tight_layout()
    fig.savefig(out("noise_sweep_scores"))
    print(f"saved {{pushforward_noise_sweep, noise_sweep_scores}}_{tag}.pdf in {out_dir}")
    return cases


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.05, help="relative snapshot noise: std of each variable = noise x its clean training range "
                         "(0.05 = 5%%)")
    ap.add_argument("--noise_ref", default="pooled", choices=["pooled", "trajectory"],
                    help="range the relative noise refers to: pooled (default) = each variable's range over all "
                         "trajectories, the same std for every trajectory; trajectory = each trajectory's own range")
    ap.add_argument("--sweep", type=float, nargs="*", default=None,
                    help="sweep these noise levels instead of a single --noise run and compare the push-forward "
                         "results; with no values: 0, 0.01, ..., 0.10, 0.15, 0.20")
    ap.add_argument("--T_train", type=float, default=6.0, help="training window [0, T_train]")
    ap.add_argument("--T_test", type=float, default=12.0, help="test window (T_train, T_test]")
    ap.add_argument("--clean_ic", action="store_true",
                    help="noise-free first snapshot of every trajectory: the initial condition is known exactly "
                         "(fits and rollouts start from it); all other snapshots stay noisy")
    ap.add_argument("--n_traj", type=int, default=20,
                    help="number of training trajectories, each from its own random initial condition (default 20)")
    ap.add_argument("--K", type=int, default=None,
                    help="training snapshots per trajectory (default: keep dt = 6/299)")
    ap.add_argument("--n_pf", type=int, default=500,
                    help="posterior draws pushed forward (at most 8000 = 4 chains x 2000)")
    ap.add_argument("--system", default="damped", choices=list(SYSTEMS),
                    help="true ROM generating the data (defined in toy_systems.py)")
    ap.add_argument("--deriv", default="dbr", choices=["dbr", "sg", "fd"],
                    help="derivative estimates: De Brabanter (Guo et al.), Savitzky-Golay, or finite differences")
    ap.add_argument("--k_dbr", default="auto",
                    type=lambda s: s if s in ("auto", "K30") else int(s),
                    help="number of De Brabanter terms: auto (default) = min(plug-in bias-variance choice, K // 30); "
                         "K30 = K // 30 as in Guo et al.; or an integer")
    ap.add_argument("--dbr_edge", default="trim", choices=["trim", "guo", "onesided"],
                    help="De Brabanter boundary: trim (default) drops the k edge snapshots; guo keeps all snapshots "
                         "with ZERO end derivatives as their code; onesided uses one-sided end derivatives")
    ap.add_argument("--lam", type=float, default=None,
                    help="fixed lam for all rows, penalty lam^2 (default: per-row fixed point from lam = 50)")
    ap.add_argument("--prior", default="tikhonov", choices=list(PRIORS),
                    help="prior of MCMC ONLY; the analytical posterior always uses tikhonov (fixed point or --lam). "
                         "tikhonov: Gamma_i = lam_i^2 I; wide: Gaussian, Gamma = 1e-8 I; flat: improper flat; "
                         "uniform: U(-b, b) on every entry, b = --uniform_bound; laplace / cauchy / student_t: "
                         "independent per entry, scale --prior_scale; horseshoe: regularized (--hs_p0, --slab_scale)")
    ap.add_argument("--uniform_bound", type=float, default=10.0,
                    help="half-width b of the uniform prior U(-b, b) (prior = uniform only)")
    ap.add_argument("--prior_scale", type=float, default=1.0,
                    help="scale b of the laplace / cauchy / student_t prior")
    ap.add_argument("--prior_df", type=float, default=3.0,
                    help="degrees of freedom nu of the student_t prior (nu = 1 is Cauchy)")
    ap.add_argument("--hs_p0", type=float, default=3.0,
                    help="horseshoe: guessed number of nonzero operator entries per row (sets the global scale tau0)")
    ap.add_argument("--slab_scale", type=float, default=2.0,
                    help="horseshoe: slab scale s; entries up to ~s escape shrinkage, larger ones are regularized")
    args = ap.parse_args()
    settings = dict(T_train=args.T_train, T_test=args.T_test, K=args.K, n_pf=args.n_pf,
                    system=args.system, deriv=args.deriv, lam=args.lam,
                    k_dbr=None if args.k_dbr == "K30" else args.k_dbr, dbr_edge=args.dbr_edge,
                    prior=args.prior, uniform_bound=args.uniform_bound,
                    prior_scale=args.prior_scale, prior_df=args.prior_df,
                    hs_p0=args.hs_p0, slab_scale=args.slab_scale, noise_ref=args.noise_ref, n_traj=args.n_traj,
                    clean_ic=args.clean_ic)
    
    if args.sweep is not None:
        noise_sweep(noises=args.sweep or NOISE_SWEEP, **settings)
    
    else:   # no sweep: single run at the given noise level
        main(noise=args.noise, **settings)
