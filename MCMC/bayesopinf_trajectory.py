"""
Derivative-based vs trajectory-based likelihood for Bayesian OpInf.

Uses the same toy data as bayesopinf_mcmc_check.py (put both files in one folder).

DERIVATIVE-BASED (Guo et al., closed form, row-wise):
    r_i = finite-difference derivative estimates, D = [1, q, q⊗q] from noisy snapshots
    r_i | o_i ~ N(D o_i, sigma_i^2 I),  o_i ~ N(0, sigma_i^2 Gamma^-1)
    -> o_i | data ~ N(mu_i, Sigma_i)            (exact draws, no MCMC)

TRAJECTORY-BASED (single shooting, NUTS, all rows jointly):
    unknowns    theta = (O, s_1..s_L, sigma_y)       s_l = initial state of trajectory l
    model       q_l(t; O, s_l) solves dq/dt = O d(q),  q_l(0) = s_l   (RK4, lax.scan)
    likelihood  y_{l,k} | theta ~ N(q_l(t_k; O, s_l), sigma_y^2 I)    for all l, k
    priors      O_ij ~ N(0, prior_scale^2),  s_l ~ N(y_{l,0}, 1),  sigma_y ~ HalfNormal(0.1)

Usage:
    python bayesopinf_trajectory.py                 # rich regime
    python bayesopinf_trajectory.py --regime scarce
"""

import argparse
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
from scipy.optimize import minimize
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bayesopinf_mcmc_check as base    # toy problem: build_problem, rom_rhs, pushforward

numpyro.set_host_device_count(4)
r = base.r

REGIMES = {
    "scarce": dict(noise=0.01, T_train=4.0, K=120, n_traj=1),
    "rich":   dict(noise=0.005, T_train=6.0, K=300, n_traj=5),
}


# ----------------------------------------------------------------------------
# ROM right-hand side and a differentiable RK4 integrator in JAX
# ----------------------------------------------------------------------------
IU, JU = np.triu_indices(r)            # compact Kronecker indices (static)


def rhs(O, q):
    """f(q; O) = c + A q + H (q kron_prod q) = O d(q)."""
    d_q = jnp.concatenate([jnp.ones(1), q, q[IU] * q[JU]])
    return O @ d_q


def simulate(O, q0, dt, n_steps, substeps):
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



# Trajectory-likelihood model (all rows of O are coupled through the ODE)

def trajectory_model(Y, dt, d, prior_scale, substeps):
    """Y: n_traj x K x r noisy snapshots on a uniform time grid."""
    n_traj, K, _ = Y.shape
    O = numpyro.sample("O", dist.Normal(0.0, prior_scale).expand([r, d]).to_event(2))
    s = numpyro.sample("s", dist.Normal(Y[:, 0, :], 1.0).to_event(2))       # initial states
    sigma_y = numpyro.sample("sigma_y", dist.HalfNormal(0.1))

    Q = jax.vmap(lambda s_l: simulate(O, s_l, dt, K, substeps))(s)          # n_traj x K x r
    # an O that blows up gives inf/nan: map to a huge misfit so NUTS rejects it
    Q = jnp.where(jnp.isfinite(Q), Q, 1e6)
    numpyro.sample("y", dist.Normal(Q, sigma_y).to_event(3), obs=Y)


def laplace_warm_start(Y, dt, d, O_init, prior_scale, substeps):
    """MAP by L-BFGS in NumPyro's unconstrained space, then the inverse Hessian there.

    The MAP is the (regularized) deterministic trajectory-OpInf estimate. The inverse
    Hessian is a Laplace approximation of the posterior covariance; handing it to NUTS
    as the initial mass matrix removes most of the warmup cost.
    """
    kw = dict(Y=jnp.asarray(Y), dt=dt, d=d, prior_scale=prior_scale, substeps=substeps)
    info = initialize_model(
        jax.random.PRNGKey(0), trajectory_model, model_kwargs=kw,
        init_strategy=init_to_value(values=dict(O=jnp.asarray(O_init),
                                                s=jnp.asarray(Y[:, 0, :]), sigma_y=0.01)))
    z0 = info.param_info.z
    flat0, unravel = ravel_pytree(z0)                    # order: O, s, sigma_y (sorted)
    U = lambda x: info.potential_fn(unravel(x))
    vg = jax.jit(jax.value_and_grad(U))

    def fun(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float)

    hess = jax.jit(jax.hessian(U))
    t0 = time.time()
    
    # Trust-region Newton with the exact Hessian (46 x 46 here). L-BFGS stalls because the
    # parameters live on very different scales (initial states ~1e-3, operators ~1e-1).
    opt = minimize(fun, np.asarray(flat0), jac=True, method="trust-exact",
                   hess=lambda x: np.asarray(hess(jnp.asarray(x)), dtype=float),
                   options=dict(maxiter=200, gtol=1e-6))
    
    Hs = np.asarray(hess(jnp.asarray(opt.x)))
    Hs = 0.5 * (Hs + Hs.T)
    w, V = np.linalg.eigh(Hs)
    Hinv = (V / np.clip(w, 1e-8 * w.max(), None)) @ V.T  # guard against non-PD directions
    z_map = unravel(jnp.asarray(opt.x))
    map_vals = {k: v for k, v in info.postprocess_fn(z_map).items() if k in ("O", "s", "sigma_y")}
    print(f"MAP: {opt.nit} trust-region Newton iterations, {time.time() - t0:.1f}s, "
          f"|grad| = {np.linalg.norm(opt.jac):.1e}, sigma_y at MAP = {float(map_vals['sigma_y']):.4f}")
    return map_vals, Hinv


def run_trajectory_based_likelihood_method(Y, dt, O_init, prior_scale=1.0, substeps=1,
                   num_warmup=500, num_samples=1000, seed=0):
    d = O_init.shape[1]
    map_vals, Hinv = laplace_warm_start(Y, dt, d, O_init, prior_scale, substeps)
    
    # Keep the Laplace mass matrix FIXED. NumPyro's adaptation regularizes the estimated
    # covariance toward ~1e-3/(n+5) * I, which swamps the tiny posterior variances of the
    # initial states (~1e-7) and collapses the step size. Only the step size is adapted.
    kernel = NUTS(trajectory_model, dense_mass=True, inverse_mass_matrix=jnp.asarray(Hinv),
                  adapt_mass_matrix=False, step_size=0.3, target_accept_prob=0.9,
                  init_strategy=init_to_value(values=map_vals), max_tree_depth=8)
    
    ###
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=4, progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), Y=jnp.asarray(Y), dt=dt, d=d,
             prior_scale=prior_scale, substeps=substeps,
             extra_fields=("diverging", "num_steps"))
    smp = mcmc.get_samples(group_by_chain=True)
    
    O_c = np.asarray(smp["O"])            # chains x draws x r x d
    elapsed = time.time() - t0            # after pulling samples: JAX dispatch is async
    ef = mcmc.get_extra_fields()
    
    diag = dict(
        div=int(np.sum(ef["diverging"])),
        steps=float(np.mean(ef["num_steps"])),
        ess_min=float(np.min(effective_sample_size(O_c.reshape(*O_c.shape[:2], -1)))),
        rhat_max=float(np.max(split_gelman_rubin(O_c.reshape(*O_c.shape[:2], -1)))),
        time=elapsed,
    )
    return O_c.reshape(-1, r, d), np.asarray(smp["sigma_y"]).reshape(-1), map_vals, diag


# ----------------------------------------------------------------------------
# Scoring (same metrics as bayesopinf_priors.py)
# ----------------------------------------------------------------------------
def score_operators(O_samp, O_true):
    mean = O_samp.mean(0)
    lo, hi = np.percentile(O_samp, [2.5, 97.5], axis=0)
    return dict(rel_err=np.linalg.norm(mean - O_true) / np.linalg.norm(O_true),
                coverage=np.mean((O_true >= lo) & (O_true <= hi)),
                width=np.mean(hi - lo),
                zero_abs=np.mean(np.abs(mean[O_true == 0])))


def score_pushforward(O_samp, prob, t_pred, n=300, seed=1):
    rng = np.random.default_rng(seed)
    Y = base.pushforward(O_samp[rng.choice(len(O_samp), n, replace=False)], t_pred, prob["q0"])
    truth = solve_ivp(base.rom_rhs(prob["O_true"]), (t_pred[0], t_pred[-1]), prob["q0"],
                      t_eval=t_pred, rtol=1e-10, atol=1e-12).y
    lo, hi = np.percentile(Y, [2.5, 97.5], axis=0)
    return Y, truth, dict(finite=len(Y) / n,
                          traj_cov=np.mean((truth >= lo) & (truth <= hi)),
                          band_w=np.mean(hi - lo))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=REGIMES, default="rich")
    ap.add_argument("--prior-scale", type=float, default=1.0,
                    help="std of the N(0, s^2) prior on each operator entry (trajectory method)")
    ap.add_argument("--substeps", type=int, default=1, help="RK4 steps per snapshot interval")
    ap.add_argument("--horizon", type=float, default=2.0)
    args = ap.parse_args()

    cfg = REGIMES[args.regime]
    prob = base.build_problem(**cfg)
    O_true, t = prob["O_true"], prob["t"]
    dt = t[1] - t[0]
    n_traj, K = cfg["n_traj"], cfg["K"]
    # recover the noisy snapshots from D = [1, Q^T, ...] (same data both methods see)
    Y = prob["D"][:, 1:1 + r].reshape(n_traj, K, r)
    t_pred = np.linspace(0, args.horizon * t[-1], 400)
    print(f"regime = {args.regime}: {n_traj} trajectories x {K} snapshots, "
          f"true snapshot noise = {cfg['noise']}\n")


    # derivative-based: exact Gaussian posterior (Guo et al.)
    rng = np.random.default_rng(0)
    n_draw = 4000
    O_deriv = np.stack([rng.multivariate_normal(prob["mu"][i], prob["Sigma"][i], n_draw)
                        for i in range(r)], axis=1)                    # draws x r x d


    # trajectory-based: NUTS, initialized at the derivative-based mean
    O_traj, sig_y, map_vals, diag = run_trajectory_based_likelihood_method(Y, dt, prob["mu"], args.prior_scale,
                                                   args.substeps)
    print(f"trajectory NUTS: {diag['div']} divergences, min ESS {diag['ess_min']:.0f}, "
          f"max R-hat {diag['rhat_max']:.3f}, {diag['steps']:.0f} leapfrog steps/iter, "
          f"{diag['time']:.0f}s")
    lo, hi = np.percentile(sig_y, [2.5, 97.5])
    print(f"  inferred snapshot noise sigma_y = {sig_y.mean():.4f}  "
          f"(95% CI [{lo:.4f}, {hi:.4f}], truth {cfg['noise']})")

    results = {}
    for name, O_s in [("derivative", O_deriv), ("trajectory", O_traj)]:
        Yp, truth, pf = score_pushforward(O_s, prob, t_pred)
        results[name] = dict(O=O_s, Y=Yp, truth=truth, **score_operators(O_s, O_true), **pf)

    hdr = (f"\n{'likelihood':>10s} | {'op rel err':>10s} | {'op 95% cov':>10s} | "
           f"{'op CI width':>11s} | {'|mean| at true 0':>16s} | {'stable':>6s} | "
           f"{'traj 95% cov':>12s} | {'band width':>10s}")
    print(hdr); print("-" * len(hdr))
    for n, s in results.items():
        print(f"{n:>10s} | {s['rel_err']:10.3f} | {s['coverage']:10.2f} | {s['width']:11.3f} | "
              f"{s['zero_abs']:16.3f} | {s['finite']:6.2f} | {s['traj_cov']:12.2f} | "
              f"{s['band_w']:10.4f}")


    # --- forest plot ---------------------------------------------------------
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
    fig.suptitle(f"Operator posteriors (95% CI), {args.regime} data")
    fig.tight_layout()
    # fig.savefig(f"traj_vs_deriv_operators_{args.regime}.png", dpi=300)
    fig.savefig(f"traj_vs_deriv_operators_{args.regime}.pdf")



    # --- prediction bands ----------------------------------------------------
    fig, axes = plt.subplots(r, 2, figsize=(9, 2.3 * r), sharex=True, sharey="row")
    for k, (n, s) in enumerate(results.items()):
        lo_, med, hi_ = np.percentile(s["Y"], [2.5, 50, 97.5], axis=0)
        for j in range(r):
            ax = axes[j, k]
            ax.fill_between(t_pred, lo_[j], hi_[j], color=f"C{k}", alpha=0.3)
            ax.plot(t_pred, med[j], color=f"C{k}", lw=1)
            ax.plot(t_pred, s["truth"][j], "k--", lw=1)
            ax.axvline(t[-1], color="0.5", lw=0.8, ls=":")
            if j == 0:
                ax.set_title(f"{n} likelihood", fontsize=10)
            if k == 0:
                ax.set_ylabel(f"$\\hat q_{j}$")
    for ax in axes[-1]:
        ax.set_xlabel("t")
    fig.suptitle("95% prediction bands (dashed: truth, dotted: end of training data)")
    fig.tight_layout()
    # fig.savefig(f"traj_vs_deriv_pushforward_{args.regime}.png", dpi=300)
    fig.savefig(f"traj_vs_deriv_pushforward_{args.regime}.pdf")
    print(f"\nsaved traj_vs_deriv_operators_{args.regime}.{{pdf}}, "
          f"traj_vs_deriv_pushforward_{args.regime}.{{pdf}}")


if __name__ == "__main__":
    main()
