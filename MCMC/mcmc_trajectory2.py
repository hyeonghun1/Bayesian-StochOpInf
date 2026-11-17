"""
Derivative-based vs trajectory-based likelihood for Bayesian OpInf.

Uses the same toy data and derivative method as mcmc_derivative_derv_fixed.py (put both files in one folder).

The derivative method is the closed-form Gaussian posterior, so it ALWAYS uses the Gaussian Tikhonov prior
(lam_i by the fixed point, or fixed --lam). --prior is the prior of the trajectory method only; with
--prior tikhonov both methods share the prior and differ only in the likelihood.

DERIVATIVE-BASED (Guo et al., closed form, row-wise; everything from mcmc_derivative_derv_fixed.build_problem):
    D = [1, q, qkronq] from the RAW noisy snapshots, r_i = derivative estimates, with the same options and
    defaults as mcmc_derivative_derv_fixed.py: --deriv dbr (De Brabanter, default) | sg | fd;
    --dbr_edge trim (default: drop the k edge snapshots) | guo (their code: zero end derivatives) | onesided;
    --k_dbr auto (default: min(plug-in bias-variance choice, K // 30)) | K30 (their code) | an integer.
    --dbr_edge guo --k_dbr K30 reproduces Guo et al.'s ddt_localpoly exactly.
    r_i | o_i ~ N(D o_i, sigma_i^2 I),  o_i ~ N(0, sigma_i^2 Gamma_i^-1)
    -> o_i | data ~ N(mu_i, Sigma_i)            (exact draws, no MCMC)
    Gamma_i = lam_i^2 I, lam_i by the Tikhonov fixed point (or fixed --lam), whatever --prior is.

TRAJECTORY-BASED (single shooting, NUTS, all rows jointly):
    unknowns    theta = (O, s_1..s_L, sigma_y)       s_l = initial state of trajectory l
    model       q_l(t; O, s_l) solves dq/dt = O d(q),  q_l(0) = s_l   (RK4, lax.scan)
    likelihood  y_{l,k} | theta ~ N(q_l(t_k; O, s_l), sigma_y^2 I)    for all l, k
                (the raw noisy snapshots; no derivatives, no noisy D)
    noise       one sigma_y per state variable (sigma_y is a vector of length r), so --noise, which
                gives each variable its own noise level, is modelled correctly
    priors      s_l ~ N(y_{l,0}, 1),  sigma_y,j ~ HalfNormal(std of the data of variable j) (weakly
                informative: the noise cannot exceed the total spread of the data), and on O (--prior):
                tikhonov -> o_i ~ N(0, sigma_i^2 / lam_i^2 I), the derivative method's own prior (its
                          plug-in sigma_i^2, lam_i);
                wide     -> N(0, 1e8 sigma_i^2 I) (noninformative Gaussian);
                flat     -> improper uniform;
                uniform  -> U(-b, b) per entry, b = --uniform_bound;
                laplace  -> Laplace(0, s), s = --prior-scale (Bayesian lasso: favours exact zeros);
                studentt -> StudentT(nu, 0, s), nu = --nu (nu = 1 is Cauchy): heavy tails, large
                            entries shrunk less

Usage:
    python mcmc_trajectory2.py                 # rich regime, De Brabanter + Guo lam, Tikhonov prior
    python mcmc_trajectory2.py --regime scarce
    python mcmc_trajectory2.py --plot_traj 5    # also predict and plot the first 5 of the 10 trajectories
    python mcmc_trajectory2.py --prior flat    # trajectory method: noninformative prior
    python mcmc_trajectory2.py --prior uniform --uniform_bound 10
    python mcmc_trajectory2.py --prior laplace --prior-scale 0.5
    python mcmc_trajectory2.py --prior studentt --nu 1 --prior-scale 1     # Cauchy
    python mcmc_trajectory2.py --deriv sg
    python mcmc_trajectory2.py --dbr_edge guo --k_dbr K30     # exactly Guo et al.'s derivatives
    python mcmc_trajectory2.py --system limit_cycle --deriv fd
    python mcmc_trajectory2.py --noise 0.05 --T_train 6 --T_test 12 --n_pf 500   # noise std = 5% of range
--noise / --T_train / --n_traj / --K override the regime's values; --T_test defaults to 2 * T_train.
--noise is relative: the std of variable j is noise x its clean training range (0.05 = 5%); the range is
over all trajectories together (--noise_ref pooled, default: the same std for every trajectory) or each
trajectory's own (--noise_ref trajectory).
Figures go to pdfs/mcmc_trajectory2_pdfs/.
"""

import argparse
import functools
import os
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import initialize_model
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize, least_squares
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mcmc_derivative_derv_fixed as base    # toy problem + derivative method: build_problem, rom_rhs, pushforward

numpyro.set_host_device_count(4)
r = base.r

PRIORS = ["tikhonov", "wide", "flat", "uniform", "laplace", "studentt", "horseshoe"]   # --prior: trajectory only
SPARSE_PRIORS = ["laplace", "studentt"]       # get a Gaussian stand-in in the least-squares warm start
WIDE_SD = 1e4                                 # "wide": prior sd = 1e4 sigma_i (Gamma = 1e-8 I)

REGIMES = {
    "scarce": dict(noise=0.01, T_train=4.0, K=300, n_traj=1),
    "rich":   dict(noise=0.01, T_train=4.0, K=300, n_traj=10),
}


# ROM right-hand side and a differentiable RK4 integrator in JAX
IU, JU = np.triu_indices(r)            # compact Kronecker indices (static)


def rhs(O, q):
    """f(q; O) = c + A q + H (q kron_prod q) = O d(q)."""
    d_q = jnp.concatenate([jnp.ones(1), q, q[IU] * q[JU]])
    return O @ d_q


def simulate_ODE(O, q0, dt, n_steps, substeps):
    """RK4 from q0 with step dt/substeps; returns the state at n_steps snapshot times."""
    h = dt / substeps

    def rk4(q, _):
        k1 = rhs(O, q)
        k2 = rhs(O, q + 0.5 * h * k1)
        k3 = rhs(O, q + 0.5 * h * k2)
        k4 = rhs(O, q + h * k3)
        return q + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None

    def snapshot_step(q, _):
        for _ in range(substeps):     # static Python loop: unrolled at trace time
            q, _ = rk4(q, None)
        return q, q

    _, qs = lax.scan(snapshot_step, q0, None, length=n_steps - 1)
    return jnp.vstack([q0[None, :], qs])                 # n_steps x r



# Prior on O of the trajectory model

def operator_prior(r, d, prior, prior_sd=None, bound=None, nu=None, hs=None):
    """Sample site "O" (r x d) with the given prior; see trajectory_model for the options."""
    if prior == "horseshoe":    # regularized horseshoe per row, non-centered: O = z * tau * lt
        tau = numpyro.sample("tau", dist.HalfCauchy(jnp.asarray(hs["tau0"])).to_event(1))          # r
        lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones((r, d))).to_event(2))                  # r x d
        c2 = numpyro.sample("c2", dist.InverseGamma(2.0 * jnp.ones(r), 2.0 * hs["slab"]**2).to_event(1))
        lt = jnp.sqrt(c2[:, None] * lam**2 / (c2[:, None] + tau[:, None]**2 * lam**2))
        z = numpyro.sample("z", dist.Normal(jnp.zeros((r, d)), 1.0).to_event(2))
        return numpyro.deterministic("O", z * tau[:, None] * lt)
    if prior == "uniform":      # u ~ U[-bound, bound]
        u = numpyro.sample("u", dist.Uniform(-jnp.ones((r, d)), jnp.ones((r, d))).to_event(2))
        return numpyro.deterministic("O", bound * u)
    if prior == "flat":
        return numpyro.sample("O", dist.ImproperUniform(
            dist.constraints.independent(dist.constraints.real, 2), (), (r, d)))
    if prior == "normal":
        return numpyro.sample("O", dist.Normal(jnp.zeros((r, d)), jnp.asarray(prior_sd)).to_event(2))
    if prior == "laplace":
        return numpyro.sample("O", dist.Laplace(jnp.zeros((r, d)), prior_sd).to_event(2))
    if prior == "studentt":
        return numpyro.sample("O", dist.StudentT(nu, jnp.zeros((r, d)), prior_sd).to_event(2))
    raise ValueError(f"unknown prior on O {prior!r}")


# Trajectory-likelihood model (all rows of O are coupled through the ODE)

def trajectory_model(Y, dt, d, substeps, prior="normal", prior_sd=None, bound=None, sigma_scale=None,
                     nu=None, hs=None):
    """Y: n_traj x K x r noisy snapshots on a uniform time grid.

    Noise: y_{l,k,j} ~ N(q_{l,j}(t_k), sigma_y,j^2), one sigma per state variable j, with prior
    sigma_y,j ~ HalfNormal(sigma_scale_j) (sigma_scale: length-r array, bound with functools.partial).

    Prior on O (bind prior / prior_sd / bound with functools.partial, so they stay static):
      "normal"  O_ij ~ N(0, prior_sd_ij^2), prior_sd an r x d array (or scalar)
      "flat"    improper uniform on R^{r x d} (no prior term)
      "uniform" O_ij ~ U(-bound, bound), sampled as u ~ U(-1, 1), O = bound * u (constant Jacobian;
                a fixed U(-1, 1) keeps the box transform free of traced arguments)
      "laplace"   O_ij ~ Laplace(0, prior_sd)
      "studentt"  O_ij ~ StudentT(nu, 0, prior_sd)
      "horseshoe" regularized horseshoe per row i (Piironen & Vehtari 2017), non-centered:
                  O_ij = z_ij tau_i lt_ij, z ~ N(0, 1), lam_ij ~ C+(0, 1), tau_i ~ C+(0, tau0_i),
                  lt^2 = c_i^2 lam^2 / (c_i^2 + tau_i^2 lam^2), c_i^2 ~ Inv-Gamma(2, 2 slab^2);
                  hs = dict(tau0=(r,) array, slab=float). The likelihood is not Gaussian in O here
                  (O enters through the ODE), so O cannot be integrated out as in the derivative script.
    For laplace / studentt prior_sd is a scalar.
    """
    n_traj, K, _ = Y.shape
    
    # Sample priors on the operator, initial states, and noise levels
    O = operator_prior(r, d, prior, prior_sd, bound, nu, hs)
    
    # sample initial states s^l ~ N(y_l0, 1) for each trajectory l
    # (since the initial state is known to within the snapshot noise)
    s = numpyro.sample("s", dist.Normal(Y[:, 0, :], 1.0).to_event(2))       
    
    # one noise level per state variable, with a weakly informative prior
    # (the noise cannot exceed the total)
    scale = jnp.full(r, 0.1) if sigma_scale is None else jnp.asarray(sigma_scale)
    sigma_y = numpyro.sample("sigma_y", dist.HalfNormal(scale).to_event(1)) # sample noise levels (r,)

    # q_k^l = RK4 solution from s^l with operator O
    # jax.vmap over the trajectories: each trajectory has its own initial state s^l, but the same O
    # This serves as the likelihood mean for the noisy snapshots Y.
    Q = jax.vmap(lambda s_l: simulate_ODE(O, s_l, dt, K, substeps))(s)          # n_traj x K x r
    
    # An O that blows up gives inf/nan: map to a huge misfit so NUTS rejects it
    Q = jnp.where(jnp.isfinite(Q), Q, 1e6)
    
    # likelihood: p(y_{l,k,j} | O, s^l, sigma_{y,j}) = N(q_{l,k,j}, sigma_{y,j}^2)
    numpyro.sample("y", dist.Normal(Q, sigma_y).to_event(3), obs=Y)        # sigma_y broadcasts over r


def init_operator_latents(prior, O, bound=None, hs=None):
    """Values of the latent operator sites of trajectory_model that give the operator O."""
    O = np.asarray(O, dtype=float)
    if prior == "uniform":
        return dict(u=jnp.asarray(O / bound))
    if prior == "horseshoe":
        # tau = tau0, c = slab, and per entry the local scale whose prior sd tau * lt equals |O_ij|
        # (at least 1e-3, below 0.9 slab), so z = O / (tau lt) = +-1 for the entries that matter
        tau = np.asarray(hs["tau0"], dtype=float)
        c2 = np.full(O.shape[0], hs["slab"]**2)
        v = np.clip(np.abs(O), 1e-3, 0.9 * hs["slab"])                     # target prior sd tau * lt
        lt = v / tau[:, None]
        lam = np.sqrt(lt**2 * c2[:, None] / (c2[:, None] - tau[:, None]**2 * lt**2))
        return dict(tau=jnp.asarray(tau), lam=jnp.asarray(lam), c2=jnp.asarray(c2), z=jnp.asarray(O / v))
    return dict(O=jnp.asarray(O))


def estimate_noise(Y):
    """Per-variable snapshot-noise estimate (length r) from second differences:
    y_{k+1} - 2 y_k + y_{k-1} ~ N(0, 6 sigma_j^2) plus a smooth part of size |q''| dt^2 (negligible
    for the sampling rates used here). Median absolute deviation, so a few large residuals do not matter."""
    d2 = Y[:, 2:, :] - 2.0 * Y[:, 1:-1, :] + Y[:, :-2, :]
    return np.median(np.abs(d2), axis=(0, 1)) / 0.6745 / np.sqrt(6.0)


def laplace_warm_start(model, Y, dt, d, O_init, substeps, prior="normal", prior_sd=None, bound=None,
                       n_starts=3, hs=None):
    """Robust MAP + Laplace approximation, in three stages.

    1. sigma_y is fixed at a data-based noise estimate (second differences), and (O, s) are fitted by
       nonlinear least squares (scipy trf with the exact JAX Jacobian of the scaled residuals), from
       several starts: the derivative-based mean, the same with H = 0, and half of it. The best fit wins.
       Least squares on residuals is far more robust than Newton on the log-posterior from a poor start.
    2. Trust-region Newton on the full log-posterior (O or u, s, sigma_y) in NumPyro's unconstrained
       space, started from stage 1 (sigma_y = RMS residual). Kept only if it lowers the potential.
    3. Inverse Hessian at the MAP = Laplace covariance, used as the fixed NUTS mass matrix and to
       draw dispersed chain starts.
    Returns the MAP (constrained values), the inverse Hessian, the unconstrained MAP vector, the unravel
    function and a status dict. Prints warnings if a stage does not converge.
    """
    Y = np.asarray(Y)
    n_traj, K, _ = Y.shape
    n_O = r * d
    sigma0 = estimate_noise(Y)                         # (r,)
    sig0j = jnp.asarray(sigma0)                        # broadcasts over the last axis
    Yj, Y0 = jnp.asarray(Y), jnp.asarray(Y[:, 0, :])
    t0 = time.time()

    # stage 1 and 2 finds the MAP \theta^*

    # stage 1: least squares in x = [vec(O), vec(s)] with sigma_y = sigma0 fixed
    def residuals(x):
        O = x[:n_O].reshape(r, d)
        s_ = x[n_O:].reshape(n_traj, r)
        Q = jax.vmap(lambda s_l: simulate_ODE(O, s_l, dt, K, substeps))(s_)
        Q = jnp.where(jnp.isfinite(Q), Q, 1e3)
        parts = [((Yj - Q) / sig0j).ravel(), (s_ - Y0).ravel()]           # s_l ~ N(y_l0, 1)
        if prior in ["normal", "horseshoe"] + SPARSE_PRIORS:
            # O_ij ~ N(0, sd_ij^2); for laplace / studentt / horseshoe a Gaussian stand-in of the
            # same scale (horseshoe: the slab scale) (without it the fit is unregularized and drifts to unstable operators);
            # stage 2 then uses the exact prior
            parts.append((O / jnp.asarray(prior_sd)).ravel())
        return jnp.concatenate(parts)
    
    res_j = jax.jit(residuals)
    jac_j = jax.jit(jax.jacfwd(residuals))  # Jacobian of the residuals for least squares
    lsq_bounds = (-np.inf, np.inf)
    
    if prior == "uniform":                                                  # O inside the box
        lo = np.concatenate([np.full(n_O, -0.999 * bound), np.full(n_traj * r, -np.inf)])
        lsq_bounds = (lo, -lo)

    O_init = np.asarray(O_init, dtype=float)
    O_noH = O_init.copy()
    O_noH[:, 1 + r:] = 0.0
    
    starts = [O_init, O_noH, 0.5 * O_init][:max(1, n_starts)]
    best = None
    
    for k, O0 in enumerate(starts):
        if prior == "uniform":
            O0 = np.clip(O0, -0.99 * bound, 0.99 * bound)
        x0 = np.concatenate([O0.ravel(), Y[:, 0, :].ravel()])
        # a start whose ROM blows up gives huge residuals and non-finite Jacobian entries: clean them,
        # silence the overflow warnings, and simply skip the start if the solver still fails
        f = lambda x: np.nan_to_num(np.asarray(res_j(jnp.asarray(x))), nan=1e6, posinf=1e6, neginf=-1e6)
        J = lambda x: np.nan_to_num(np.asarray(jac_j(jnp.asarray(x))), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            with np.errstate(all="ignore"):
                sol = least_squares(f, x0, jac=J, method="trf", x_scale="jac", bounds=lsq_bounds,
                                    max_nfev=300, ftol=1e-12, xtol=1e-12, gtol=1e-10)
        except Exception:
            continue
        if np.isfinite(sol.cost) and (best is None or sol.cost < best.cost):
            best, best_k = sol, k
            
    if best is None:
        raise RuntimeError("trajectory MAP: every least-squares start failed (the ROM blows up from all "
                           "starts); try another --deriv / --dbr_edge for the starting operators")
    
    x1 = best.x
    misfit = np.asarray(res_j(jnp.asarray(x1)))[: n_traj * K * r].reshape(n_traj, K, r) * sigma0  # y - q
    sigma1 = np.sqrt(np.mean(misfit**2, axis=(0, 1)))                           # per-variable RMS residual
    fmt = lambda a: np.array2string(np.asarray(a), precision=4)
    
    stage1 = (f"least squares: best of {len(starts)} starts = start {best_k}, {best.nfev} evaluations, "
              f"status {best.status}, RMS residual {fmt(sigma1)} (noise estimate {fmt(sigma0)})")

    # stage 2: Newton on the full potential in NumPyro's unconstrained space
    O1, s1 = x1[:n_O].reshape(r, d), x1[n_O:].reshape(n_traj, r)
    init_O = init_operator_latents(prior, O1, bound, hs)
    kw = dict(Y=Yj, dt=dt, d=d, substeps=substeps)
    info = initialize_model(
        jax.random.PRNGKey(0), model, model_kwargs=kw,
        init_strategy=init_to_value(values=dict(**init_O, s=jnp.asarray(s1), sigma_y=jnp.asarray(sigma1))))
    
    flat1, unravel = ravel_pytree(info.param_info.z)     # order: O (or u), s, sigma_y (sorted)
    U = lambda x: info.potential_fn(unravel(x))
    vg = jax.jit(jax.value_and_grad(U))
    hess = jax.jit(jax.hessian(U))     # Hessian of the potential in the unconstrained space (for Laplace covariance)

    # a trial step where the ROM blows up gives an inf potential and a nan Hessian; trust-exact builds
    # the Hessian of every trial point before rejecting it, so map those to a huge value (step rejected)
    def fun(x):
        v, g = vg(jnp.asarray(x))
        if not (np.isfinite(v) and np.all(np.isfinite(g))):
            return 1e20, np.zeros_like(np.asarray(x, dtype=float))
        return float(v), np.asarray(g, dtype=float)

    U1 = fun(np.asarray(flat1))[0]
    
    opt = minimize(fun, np.asarray(flat1), jac=True, method="trust-exact",
                   hess=lambda x: np.nan_to_num(np.asarray(hess(jnp.asarray(x)), dtype=float),
                                                nan=0.0, posinf=0.0, neginf=0.0),
                   options=dict(maxiter=100, gtol=1e-6))
    
    x_map = opt.x if (np.isfinite(opt.fun) and opt.fun <= U1) else np.asarray(flat1)
    g_map = fun(x_map)[1]

    # stage 3: Laplace covariance H^{-1} near the MAP (for NUTS mass matrix and dispersed chain starts)
    Hs = np.asarray(hess(jnp.asarray(x_map)))
    Hs = 0.5 * (Hs + Hs.T)
    w, V = np.linalg.eigh(Hs)
    Hinv = (V / np.clip(w, 1e-8 * w.max(), None)) @ V.T  # guard against non-PD directions
    sites = tuple(info.param_info.z)                     # latent sites: O (or its latents), s, sigma_y
    map_vals = {k: v for k, v in info.postprocess_fn(unravel(jnp.asarray(x_map))).items() if k in sites}
    sig_map = np.asarray(map_vals["sigma_y"])
    status = dict(newton_success=bool(opt.success), grad=float(np.linalg.norm(g_map)),
                  min_eig=float(w.min()), sigma_map=sig_map, sigma_est=sigma0)
    print(f"MAP stage 1 ({stage1})")
    print(f"MAP stage 2: {opt.nit} trust-region Newton iterations ({'converged' if opt.success else 'NOT converged'}), "
          f"{time.time() - t0:.1f}s total, |grad| = {status['grad']:.1e}, sigma_y at MAP = {fmt(sig_map)}, "
          f"min Hessian eigenvalue = {w.min():.2e}")
    if status["grad"] > 1e-2:
        print("  WARNING: MAP gradient is not small; NUTS may start far from the posterior mode")
    if w.min() <= 0:
        print("  WARNING: Hessian at the MAP is not positive definite (saddle or flat direction)")
    if np.any(sig_map > 1.5 * sigma0):
        print(f"  WARNING: sigma_y at MAP ({fmt(sig_map)}) is well above the data noise estimate ({fmt(sigma0)}): "
              "the fitted trajectories do not pass through the data (wrong local optimum?)")
    return map_vals, Hinv, x_map, unravel, status


# Run the trajectory-based likelihood method with NUTS,
# returning the posterior draws of O and sigma_y, the MAP, and a diagnostics dict
def run_trajectory_based_likelihood_method(Y, dt, O_init, prior="normal", prior_sd=1.0, bound=None,
                                           substeps=1, num_warmup=500, num_samples=1000, seed=0,
                                           nu=None, hs=None):
    d = O_init.shape[1]
    
    # weakly informative noise prior: sigma_y,j ~ HalfNormal(std of all snapshots of variable j)
    sigma_scale = np.asarray(Y).reshape(-1, r).std(axis=0)
    
    # model w/ fixed options
    model = functools.partial(trajectory_model, prior=prior,
                              prior_sd=None if prior_sd is None else np.asarray(prior_sd, dtype=float),
                              bound=None if bound is None else float(bound), sigma_scale=sigma_scale,
                              nu=None if nu is None else float(nu), hs=hs)
    
    # warm start: MAP + Laplace covariance (inverse Hessian) for the NUTS mass matrix
    map_vals, Hinv, x_map, unravel, ws_status = laplace_warm_start(
        model, Y, dt, d, O_init, substeps, prior=prior,
        prior_sd=(np.full((r, d), hs["slab"]) if prior == "horseshoe"        # Gaussian stand-in for stage 1
                  else None if prior_sd is None else np.broadcast_to(np.asarray(prior_sd, dtype=float), (r, d))),
        bound=bound if prior == "uniform" else None, hs=hs)

    # Dispersed chain starts: draw each chain's start from the Laplace approximation N(x_map, Hinv)
    # (in the unconstrained space). If the MAP was a poor local optimum or the Laplace covariance is
    # wrong, the chains now disagree and R-hat flags it; identical starts would hide it.
    num_chains = 4
    L = np.linalg.cholesky(Hinv + 1e-14 * np.eye(len(x_map)))
    xi = np.random.default_rng(seed).standard_normal((num_chains, len(x_map)))
    # A start where the ROM blows up has a potential ~1e16 and leaves its chain stuck, diverging at every
    # step (common for the horseshoe, whose scale directions are wide). Shrink such a start toward the MAP
    # until its potential is plausible for a draw from the Laplace approximation: a draw sits ~dim/2
    # above the MAP, so accept U - U_map <= dim.
    info = initialize_model(jax.random.PRNGKey(0), model,
                            model_kwargs=dict(Y=jnp.asarray(Y), dt=dt, d=d, substeps=substeps),
                            init_strategy=init_to_value(values=map_vals))
    U = lambda x: float(info.potential_fn(unravel(jnp.asarray(x))))
    U0, dim = U(x_map), len(x_map)
    starts, shrink = [], []
    for c in range(num_chains):
        f = 1.0
        while f > 1e-3 and not U(x_map + f * (L @ xi[c])) - U0 <= dim:
            f *= 0.5
        shrink.append(f)
        starts.append(unravel(jnp.asarray(x_map + f * (L @ xi[c]))))
    if min(shrink) < 1.0:
        print(f"  chain starts shrunk toward the MAP by {np.array2string(np.array(shrink), precision=3)} "
              "(full Laplace draws had implausible potentials)")
    init_params = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *starts)
    
    # Keep the Laplace mass matrix FIXED. NumPyro's adaptation regularizes the estimated
    # covariance toward ~1e-3/(n+5) * I, which swamps the tiny posterior variances of the
    # initial states (~1e-7) and collapses the step size. Only the step size is adapted.
    # Exception: the horseshoe. Its posterior over (z, log lam, log tau, log c) is far from Gaussian
    # (the scales and z trade off), so one Laplace matrix fits it badly; there the mass matrix is adapted
    # (started from the Laplace one) during a doubled warmup.
    adapt = prior == "horseshoe"
    
    if adapt:
        num_warmup *= 2
    kernel = NUTS(model, dense_mass=True, inverse_mass_matrix=jnp.asarray(Hinv),
                  adapt_mass_matrix=adapt, step_size=0.3, target_accept_prob=0.95 if adapt else 0.9,
                  init_strategy=init_to_value(values=map_vals), max_tree_depth=8)
    
    
    ## Run MCMC ##
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), Y=jnp.asarray(Y), dt=dt, d=d, substeps=substeps,
             init_params=init_params, extra_fields=("diverging", "num_steps"))
    mcmc_samples = mcmc.get_samples(group_by_chain=True)
    
    O_c = np.asarray(mcmc_samples["O"])         # chains x draws x r x d
    elapsed = time.time() - t0                  # after pulling samples: JAX dispatch is async
    ef = mcmc.get_extra_fields()
    
    diag = dict(
        div=int(np.sum(ef["diverging"])),
        steps=float(np.mean(ef["num_steps"])),
        ess_min=float(np.min(effective_sample_size(O_c.reshape(*O_c.shape[:2], -1)))),
        rhat_max=float(np.max(split_gelman_rubin(O_c.reshape(*O_c.shape[:2], -1)))),
        time=elapsed,
        **ws_status,
    )
    return O_c.reshape(-1, r, d), np.asarray(mcmc_samples["sigma_y"]).reshape(-1, r), map_vals, diag


# Scoring (same metrics as bayesopinf_priors.py)
def score_operators(O_samp, O_true):
    mean = O_samp.mean(0)
    lo, hi = np.percentile(O_samp, [2.5, 97.5], axis=0)
    return dict(rel_err=np.linalg.norm(mean - O_true) / np.linalg.norm(O_true),
                coverage=np.mean((O_true >= lo) & (O_true <= hi)),
                width=np.mean(hi - lo),
                zero_abs=np.mean(np.abs(mean[O_true == 0])))


def score_pushforward(O_samp, prob, t_pred, n=300, seed=1, l=0):
    """Push n posterior draws through the ROM from the first noisy snapshot of trajectory l and score
    the bands against that trajectory's clean truth (l = 0: the trajectory of the main figures)."""
    n = min(n, len(O_samp))
    rng = np.random.default_rng(seed)
    q0 = prob["Qs"][l][:, 0]                          # noisy IC of trajectory l (= prob["q0"] for l = 0)
    q0_ref = prob["Qs_clean_full"][l][:, 0]           # clean IC of trajectory l
    Y = base.pushforward(O_samp[rng.choice(len(O_samp), n, replace=False)], t_pred, q0)
    
    # true O from the noisy IC
    truth = solve_ivp(base.rom_rhs(prob["O_true"]), (t_pred[0], t_pred[-1]), q0,
                      t_eval=t_pred, rtol=1e-10, atol=1e-12).y          
    
    # actual trajectory l
    truth_clean = solve_ivp(base.rom_rhs(prob["O_true"]), (t_pred[0], t_pred[-1]), q0_ref,
                            t_eval=t_pred, rtol=1e-10, atol=1e-12).y    
    
    if len(Y) == 0:                   # every draw blew up: nothing to score
        return Y, truth, truth_clean, dict(finite=0.0, traj_cov=np.nan, band_w=np.nan)
    lo, hi = np.percentile(Y, [2.5, 97.5], axis=0)
    return Y, truth, truth_clean, dict(finite=len(Y) / n,
                          traj_cov=np.mean((truth_clean >= lo) & (truth_clean <= hi)),
                          band_w=np.mean(hi - lo))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=REGIMES, default="rich")
    ap.add_argument("--prior", default="tikhonov", choices=PRIORS,
                    help="prior on the operators of the TRAJECTORY method (the derivative method always uses "
                         "tikhonov): tikhonov (Gamma_i = lam_i^2 I, as the derivative method), wide (noninformative "
                         "Gaussian), flat (improper uniform), uniform (U(-b, b)), laplace (Laplace(0, s)), "
                         "studentt (StudentT(nu, 0, s)), s = --prior-scale, horseshoe (regularized; --hs_p0, "
                         "--slab_scale)")
    ap.add_argument("--uniform_bound", type=float, default=10.0,
                    help="half-width b of the uniform prior (prior = uniform only)")
    ap.add_argument("--lam", type=float, default=None,
                    help="fixed lam for all rows of the derivative method (and the trajectory method with "
                         "--prior tikhonov); default: Tikhonov fixed point from lam = 50")
    ap.add_argument("--prior-scale", type=float, default=1.0,
                    help="scale s of the laplace / studentt prior on every entry of O")
    ap.add_argument("--nu", type=float, default=3.0,
                    help="degrees of freedom of the Student-t prior (prior = studentt only; 1 = Cauchy)")
    ap.add_argument("--hs_p0", type=float, default=3.0,
                    help="horseshoe: guessed number of nonzero operator entries per row (sets tau0, as in "
                         "mcmc_derivative_derv_fixed.py)")
    ap.add_argument("--slab_scale", type=float, default=2.0,
                    help="horseshoe: slab scale; entries up to ~slab escape shrinkage")
    ap.add_argument("--substeps", type=int, default=1, help="RK4 steps per snapshot interval")
    ap.add_argument("--plot_traj", type=int, default=1,
                    help="number of training trajectories to predict and plot (the first N of n_traj); "
                         "N > 1 adds the figure traj_vs_deriv_pushforward_multi with one column per trajectory")
    ap.add_argument("--noise", type=float, default=None,
                    help="relative snapshot noise: std of each variable = noise x its range over the clean "
                         "training data, e.g. 0.05 = 5%% (default: the regime's value)")
    ap.add_argument("--clean_ic", action="store_true",
                    help="noise-free first snapshot of every trajectory: the initial condition is known exactly "
                         "(both methods' rollouts start from it, and the trajectory prior s ~ N(y_0, 1) is centred "
                         "on it); all other snapshots stay noisy")
    ap.add_argument("--noise_ref", default="pooled", choices=["pooled", "trajectory"],
                    help="range the relative noise refers to: pooled (default) = each variable's range over all "
                         "trajectories, the same std for every trajectory; trajectory = each trajectory's own range")
    ap.add_argument("--T_train", type=float, default=None,
                    help="training window [0, T_train] (default: the regime's value)")
    ap.add_argument("--n_traj", type=int, default=None,
                    help="number of training trajectories (default: the regime's value, scarce 1 / rich 10)")
    ap.add_argument("--T_test", type=float, default=None,
                    help="predict on [0, T_test], extrapolating past T_train (default: 2 * T_train)")
    ap.add_argument("--K", type=int, default=None,
                    help="K: training snapshots per trajectory; None keeps the default sampling rate")
    ap.add_argument("--n_pf", type=int, default=300,
                    help="number of posterior draws pushed forward through the ROM")
    ap.add_argument("--system", default="damped", choices=list(base.SYSTEMS),
                    help="true ROM generating the data (defined in toy_systems.py)")
    # derivative options and defaults exactly as in mcmc_derivative_derv_fixed.py
    ap.add_argument("--deriv", default="dbr", choices=["dbr", "sg", "fd"],
                    help="derivative estimates for the derivative-based likelihood: De Brabanter "
                         "(Guo et al., default), Savitzky-Golay, or finite differences")
    ap.add_argument("--k_dbr", default="auto",
                    type=lambda s: s if s in ("auto", "K30") else int(s),
                    help="number of De Brabanter terms: auto (default) = min(plug-in bias-variance choice, K // 30); "
                         "K30 = K // 30 as in Guo et al.; or an integer")
    ap.add_argument("--dbr_edge", default="trim", choices=["trim", "guo", "onesided"],
                    help="De Brabanter boundary: trim (default) drops the k edge snapshots; guo keeps all snapshots "
                         "with ZERO end derivatives as their code; onesided uses one-sided end derivatives")
    args = ap.parse_args()

    cfg = dict(REGIMES[args.regime])
    if args.noise is not None:
        cfg["noise"] = args.noise
    if args.T_train is not None:
        cfg["T_train"] = args.T_train
    if args.n_traj is not None:
        cfg["n_traj"] = args.n_traj
    if args.K is not None:
        cfg["K"] = args.K

    # derivative side exactly as mcmc_derivative_derv_fixed.py:
    # D from the raw noisy snapshots, R from the derivative estimates, and always the Gaussian
    # Tikhonov prior with per-row lam_i (Guo fixed point, or fixed --lam): its posterior is closed form
    # T_test is passed only so that build_problem also returns held-out noisy observations of
    # trajectory 0 after T_train (for the figure); the training data do not depend on it.
    T_test = args.T_test if args.T_test is not None else 2.0 * cfg["T_train"]
    
    sparse = args.prior in SPARSE_PRIORS
    prob = base.build_problem(**cfg, system=args.system, deriv=args.deriv,
                              k_dbr=None if args.k_dbr == "K30" else args.k_dbr, dbr_edge=args.dbr_edge,
                              lam=args.lam, prior="tikhonov", noise_ref=args.noise_ref, clean_ic=args.clean_ic,
                              uniform_bound=args.uniform_bound, T_test=T_test, hs_p0=args.hs_p0)
    O_true, t = prob["O_true"], prob["t"]
    dt = t[1] - t[0]
    n_traj, K = cfg["n_traj"], cfg["K"]
    T_train = t[-1]
    noise_true = prob["noise_std"]                      # true snapshot noise std per variable (trajectory 0)
    if args.noise_ref == "pooled":
        noise_desc = f"{100 * cfg['noise']:g}% of range = {np.array2string(noise_true, precision=4)}"
    else:
        noise_desc = (f"{100 * cfg['noise']:g}% of each trajectory's own range = "
                      f"{np.array2string(prob['noise_stds'].min(axis=0), precision=4)} to "
                      f"{np.array2string(prob['noise_stds'].max(axis=0), precision=4)} over the trajectories")
    
    # raw noisy snapshots: what the trajectory likelihood fits (every snapshot, also the edges)
    Y = prob["Qs"].transpose(0, 2, 1)                   # n_traj x K x r
    
    t_pred = np.linspace(0, T_test, 400)
    
    print(f"\nsystem = {args.system}, regime = {args.regime}, deriv = {args.deriv}, "
          f"prior = tikhonov (derivative) / {args.prior} (trajectory), "
          f"{n_traj} trajectories x {K} snapshots, true snapshot noise = {noise_desc}, "
          f"n_pf = {args.n_pf}, prediction horizon = {T_test:.1f} (training ends at {T_train:.1f})\n")


    # derivative-based: exact Gaussian posterior (Guo et al.), Tikhonov prior
    rng = np.random.default_rng(0)
    n_draw = 4000
    rows = [rng.multivariate_normal(prob["mu"][i], prob["Sigma"][i], n_draw) for i in range(r)]
    O_deriv = np.stack(rows, axis=1)      # draws x r x d


    # prior on O for the trajectory method (--prior); the derivative method always uses tikhonov
    sc = f"{args.prior_scale:g}"
    if args.prior in ("tikhonov", "wide"):    # o_i ~ N(0, sd_i^2 I), sd_i = sigma_i / lam_i or 1e4 sigma_i
        tp = "normal"
        sd_row = np.sqrt(prob["sigma2"]) * (1.0 / prob["lam"] if args.prior == "tikhonov" else WIDE_SD)
        tp_sd = np.repeat(sd_row[:, None], O_true.shape[1], axis=1)
        tp_desc = ("N(0, sigma_i^2/lam_i^2)" if args.prior == "tikhonov" else "N(0, 1e8 sigma_i^2)") + \
                  ", prior sd per row = " + np.array2string(tp_sd[:, 0], precision=3)
    elif sparse:
        tp, tp_sd = args.prior, args.prior_scale
        tp_desc = (f"Laplace(0, {sc})" if args.prior == "laplace"
                   else f"StudentT(nu = {args.nu:g}, 0, {sc})") + " on every entry"
    elif args.prior == "horseshoe":       # tau0_i from the derivative fit, as in mcmc_derivative_derv_fixed.py
        tp, tp_sd = "horseshoe", None
        tp_desc = (f"regularized horseshoe, p0 = {args.hs_p0:g}, slab {args.slab_scale:g}, tau0 = "
                   + np.array2string(prob["tau0"], precision=3))
    else:                                  # flat or uniform
        tp, tp_sd = args.prior, None
        tp_desc = ("improper flat" if args.prior == "flat"
                   else f"U(-{args.uniform_bound:g}, {args.uniform_bound:g})")
    print(f"prior on O: derivative tikhonov N(0, sigma_i^2/lam_i^2), lam = "
          f"{np.array2string(prob['lam'], precision=3)}; trajectory {tp_desc}")


    # trajectory-based: NUTS, initialized at the derivative-based mean
    O_traj, sig_y, map_vals, diag = run_trajectory_based_likelihood_method(
        Y, dt, prob["mu"], prior=tp, prior_sd=tp_sd, bound=args.uniform_bound,
        substeps=args.substeps, nu=args.nu,
        hs=dict(tau0=np.asarray(prob["tau0"], dtype=float), slab=float(args.slab_scale)))
    print(f"trajectory NUTS: {diag['div']} divergences, min ESS {diag['ess_min']:.0f}, "
          f"max R-hat {diag['rhat_max']:.3f}, {diag['steps']:.0f} leapfrog steps/iter, "
          f"{diag['time']:.0f}s")
    
    if diag["rhat_max"] > 1.01 or diag["ess_min"] < 400:
        print("  WARNING: trajectory NUTS did not converge (R-hat > 1.01 or ESS < 400); "
              "do not trust the trajectory results of this run")
    
    lo, hi = np.percentile(sig_y, [2.5, 97.5], axis=0)                 # per variable
    for j in range(r):
        print(f"  inferred snapshot noise of q{j}: sigma_y = {sig_y[:, j].mean():.4f}  "
              f"(95% CI [{lo[j]:.4f}, {hi[j]:.4f}], truth " +
              (f"{noise_true[j]:.4f})" if args.noise_ref == "pooled" else
               f"{prob['noise_stds'][:, j].min():.4f} to {prob['noise_stds'][:, j].max():.4f} by trajectory; "
               f"the model has one sigma_y per variable)"))

    results = {}
    for name, O_s in [("derivative", O_deriv), ("trajectory", O_traj)]:
        Yp, truth, truth_clean, pf = score_pushforward(O_s, prob, t_pred, n=args.n_pf)
        results[name] = dict(O=O_s, Y=Yp, truth=truth, truth_clean=truth_clean, **score_operators(O_s, O_true), **pf)

    hdr = (f"\n{'likelihood':>10s} | {'op rel err':>10s} | {'op 95% cov':>10s} | "
           f"{'op CI width':>11s} | {'|mean| at true 0':>16s} | {'stable':>6s} | "
           f"{'traj 95% cov':>12s} | {'band width':>10s}")
    print(hdr); print("-" * len(hdr))
    
    for n, s in results.items():
        print(f"{n:>10s} | {s['rel_err']:10.3f} | {s['coverage']:10.2f} | {s['width']:11.3f} | "
              f"{s['zero_abs']:16.3f} | {s['finite']:6.2f} | {s['traj_cov']:12.2f} | "
              f"{s['band_w']:10.4f}")


    tag = (("" if args.system == "damped" else f"{base.system_tag(args.system)}_") + args.regime
           + ("" if args.deriv == "dbr" else f"_{args.deriv}")
           + ("" if args.deriv != "dbr" or args.k_dbr == "auto" else f"_k{args.k_dbr}")      # as derv_fixed
           + ("" if args.deriv != "dbr" or args.dbr_edge == "trim" else f"_{args.dbr_edge}edge")
           + ("" if args.prior == "tikhonov" else f"_{args.prior}")
           + (f"{args.uniform_bound:g}" if args.prior == "uniform" else "")
           + (f"{args.prior_scale:g}" if sparse else "")
           + (f"_nu{args.nu:g}" if args.prior == "studentt" else "")
           + (f"_p0{args.hs_p0:g}_s{args.slab_scale:g}" if args.prior == "horseshoe" else "")
           + ("" if args.lam is None else f"_lam{args.lam:g}")
           + ("" if args.noise is None else f"_noise{args.noise:g}")
           + ("" if args.noise_ref == "pooled" else "_noiseref-traj")
           + ("_cleanIC" if args.clean_ic else "")
           + ("" if args.T_train is None else f"_Ttrain{args.T_train:g}")
           + ("" if args.n_traj is None else f"_ntraj{args.n_traj}")
           + ("" if args.K is None else f"_K{args.K}")
           + ("" if args.T_test is None else f"_Ttest{args.T_test:g}")
           + ("" if args.n_pf == 300 else f"_n_pf{args.n_pf}")
           + ("" if args.substeps == 1 else f"_sub{args.substeps}"))
    
    
    # figures go to pdfs/<this file>_pdfs/, like bayesopinf_mcmc_check.py
    here, stem = os.path.split(os.path.splitext(os.path.abspath(__file__))[0])
    out_dir = os.path.join(here, "pdfs", stem + "_pdfs")
    os.makedirs(out_dir, exist_ok=True)
    out = lambda name: os.path.join(out_dir, f"{name}_{tag}.pdf")


    # suptitle of both figures: noise level, priors, number of training observations
    traj_prior = {"tikhonov": "Gaussian (Tikhonov)", "wide": "wide Gaussian", "flat": "flat",
                  "uniform": f"U(-{args.uniform_bound:g}, {args.uniform_bound:g})",
                  "laplace": f"Laplace(0, {args.prior_scale:g})",
                  "studentt": f"Student-t({args.nu:g}, 0, {args.prior_scale:g})",
                  "horseshoe": f"horseshoe (p0 = {args.hs_p0:g}, slab {args.slab_scale:g})"}[args.prior]
    info = (f"noise {100 * cfg['noise']:g}%"
            + ("" if args.noise_ref == "pooled" else " (of each trajectory's range)")
            + f"  |  prior: {traj_prior}  |  "
            f"{n_traj} traj. x {K} = {n_traj * K} training snapshots")

    # forest plot: posterior 95% CI of every operator entry, by likelihood
    d = O_true.shape[1]
    names = ["c"] + [f"A{k}" for k in range(r)] + [f"H{k}" for k in range(d - 1 - r)]
    fig, axes = plt.subplots(r, 1, figsize=(11, 2.8 * r), sharex=True)
    for i, ax in enumerate(axes):
        for k, (n, s) in enumerate(results.items()):
            x = np.arange(d) + (k - 0.5) * 0.25
            m = s["O"][:, i].mean(0)
            lo_, hi_ = np.percentile(s["O"][:, i], [2.5, 97.5], axis=0)
            ax.errorbar(x, m, yerr=[m - lo_, hi_ - m], fmt="o", ms=3, lw=1.4,
                        color=f"C{k}", label=n if i == 0 else None)
        ax.plot(np.arange(d), O_true[i], "kx", ms=8, mew=2, label="truth" if i == 0 else None)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_ylabel(f"row {i}")
    axes[-1].set_xticks(np.arange(d), names)
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    fig.suptitle(info, fontsize=10)
    fig.tight_layout()
    # fig.savefig(f"traj_vs_deriv_operators_{tag}.png", dpi=300)
    fig.savefig(out("traj_vs_deriv_operators"))


    # push-forward: prediction bands per likelihood vs truth
    fig, axes = plt.subplots(r, 2, figsize=(9, 2.3 * r), sharex=True, sharey="row")
    for k, (n, s) in enumerate(results.items()):
        if len(s["Y"]):
            lo_, med, hi_ = np.percentile(s["Y"], [2.5, 50, 97.5], axis=0)
        for j in range(r):
            ax = axes[j, k]
            if len(s["Y"]):
                ax.fill_between(t_pred, lo_[j], hi_[j], color=f"C{k}", alpha=0.3)
                ax.plot(t_pred, med[j], color=f"C{k}", lw=1)
            
            # noisy observations of trajectory 0: training (fitted) and held-out (not used)
            ax.plot(prob["t"], prob["Q"][j], ".", color="k", ms=1.5, alpha=0.6, zorder=1,
                    label="training observations" if (j == 0 and k == 0) else None)
            t_test = prob["t_full"][prob["t_full"] > T_train]
            ax.plot(t_test, prob["Q_test"][j], ".", color="0.6", ms=1.5, alpha=0.6, zorder=1,
                    label="held-out observations" if (j == 0 and k == 0) else None)
            
            # clean truth from the clean initial condition
            ax.plot(t_pred, s["truth_clean"][j], "k-", lw=1.2, zorder=3,
                    label="truth (clean IC)" if (j == 0 and k == 0) else None)
            
            # Clean truth from the noisy initial condition, which the true operator would produce
            # This is the reference for the prediction bands, so the trajectory method should match this
            ax.plot(t_pred, s["truth"][j], "k--", lw=1, zorder=3,
                    label="true operator from noisy IC" if (j == 0 and k == 0) else None)
            
            ax.axvline(T_train, color="0.5", lw=0.8, ls=":")
            
            if j == 0:
                ax.set_title(f"{n} likelihood  (stable {s['finite']:.0%})", fontsize=10)
            if k == 0:
                ax.set_ylabel(f"$\\hat q_{j}$")
    
    for ax in axes[-1]:
        ax.set_xlabel("t")
    fig.suptitle(info, fontsize=9)
    fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False, markerscale=6)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out("traj_vs_deriv_pushforward"))
    saved = ["traj_vs_deriv_operators", "traj_vs_deriv_pushforward"]

    # push-forward of several training trajectories: one column per trajectory, both likelihoods overlaid
    n_plot = min(args.plot_traj, n_traj)
    if n_plot > 1:
        multi = {name: [score_pushforward(results[name]["O"], prob, t_pred, n=args.n_pf, l=l)
                        for l in range(n_plot)] for name in results}
        hdr = (f"\n{'trajectory':>10s} | {'likelihood':>10s} | {'stable':>6s} | {'traj 95% cov':>12s} | "
               f"{'band width':>10s}")
        print(hdr); print("-" * len(hdr))
        for l in range(n_plot):
            for name in results:
                pf = multi[name][l][3]
                print(f"{l:>10d} | {name:>10s} | {pf['finite']:6.2f} | {pf['traj_cov']:12.2f} | "
                      f"{pf['band_w']:10.4f}")

        t_test = prob["t_full"][prob["t_full"] > T_train]
        fig, axes = plt.subplots(r, n_plot, figsize=(3.2 * n_plot, 2.3 * r + 0.8), sharex=True, squeeze=False)
        for l in range(n_plot):
            for k, name in enumerate(results):
                Yl, truth_l, clean_l, pf = multi[name][l]
                if len(Yl):
                    lo_, med, hi_ = np.percentile(Yl, [2.5, 50, 97.5], axis=0)
                for j in range(r):
                    ax = axes[j, l]
                    if len(Yl):
                        ax.fill_between(t_pred, lo_[j], hi_[j], color=f"C{k}", alpha=0.3,
                                        label=f"{name} likelihood 95%" if (j == 0 and l == 0) else None)
                        ax.plot(t_pred, med[j], color=f"C{k}", lw=1)
            for j in range(r):
                ax = axes[j, l]
                ax.plot(prob["t"], prob["Qs"][l][j], ".", color="k", ms=1.5, alpha=0.6, zorder=1,
                        label="training observations" if (j == 0 and l == 0) else None)
                ax.plot(t_test, prob["Q_tests"][l][j], ".", color="0.6", ms=1.5, alpha=0.6, zorder=1,
                        label="held-out observations" if (j == 0 and l == 0) else None)
                ax.plot(t_pred, clean_l[j], "k-", lw=1.2, zorder=3,
                        label="truth (clean IC)" if (j == 0 and l == 0) else None)
                ax.plot(t_pred, truth_l[j], "k--", lw=1, zorder=3,
                        label="true operator from noisy IC" if (j == 0 and l == 0) else None)
                ax.axvline(T_train, color="0.5", lw=0.8, ls=":")
                # limits from the truth and the data, so a band that blows up does not hide the rest
                data = np.concatenate([clean_l[j], prob["Qs"][l][j], prob["Q_tests"][l][j]])
                pad = 0.15 * (data.max() - data.min())
                ax.set_ylim(data.min() - pad, data.max() + pad)
                if l == 0:
                    ax.set_ylabel(f"$\\hat q_{j}$")
            stab = ", ".join(f"{dict(derivative='deriv.', trajectory='traj.')[name]} {multi[name][l][3]['finite']:.0%}"
                             for name in results)
            axes[0, l].set_title(f"trajectory {l}  (stable: {stab})", fontsize=8)
            axes[-1, l].set_xlabel("t")
        fig.suptitle(info, fontsize=9)
        fig.legend(loc="lower center", ncol=6, fontsize=8, frameon=False, markerscale=6)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(out("traj_vs_deriv_pushforward_multi"))
        saved.append("traj_vs_deriv_pushforward_multi")

    print(f"\nsaved {{{', '.join(saved)}}}_{tag}.pdf in {out_dir}")


if __name__ == "__main__":
    main()
