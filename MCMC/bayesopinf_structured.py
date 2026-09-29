"""
Structure-preserving Bayesian OpInf on the toy ROM, with the derivative-based and the
trajectory-based likelihood.

Needs bayesopinf_mcmc_check.py and bayesopinf_trajectory.py in the same folder.

Two kinds of structure, handled in two different ways:

  energy-preserving H   (EQUALITY constraint, q^T H(q ⊗ q) = 0)  -> REPARAMETRIZE
      H(q ⊗ q) = sum_k q_k S_k q with S_k skew. Linear in the new parameters, so the
      posterior geometry stays as nice as the unstructured one.

  dissipative A         (INEQUALITY constraint, sym(A) negative definite)  -> TRUNCATE
      Prior truncated to the constraint set, so
          p(theta | data, A dissipative)  ∝  p(theta | data) * 1[A dissipative],
      i.e. keeping the dissipative draws of the untruncated posterior is EXACT. Efficiency
      = posterior probability of the set (reported as "accept").
      Why not reparametrize A = J - L L^T? It works when the data already favor dissipative
      operators, but when the constraint is active (most posterior mass outside the set),
      the mass piles up at the boundary where L L^T loses rank. There the map L -> L L^T is
      degenerate, the posterior in L becomes quartic/funnel-shaped, and NUTS needs hundreds
      of leapfrog steps per iteration with divergences (tested: R-hat up to 1.7).

Structures compared: none | dissA (truncation) | epH (reparam.) | epH+dissA (both).
Priors: N(0, s^2) on c, free A/H entries, and on S.
Derivative likelihood: R ~ N(O D^T, diag(sigma_i^2)), sigma_i fixed (Guo et al. plug-in).
Trajectory likelihood: Y ~ N(q(t; O, s0), sigma_y^2), s0 and sigma_y inferred.
Sampler: MAP (trust-region Newton) -> Laplace whitening -> NUTS with dense mass adaptation.

Usage:
    python bayesopinf_structured.py                  # scarce data (where structure should matter)
    python bayesopinf_structured.py --regime rich
"""

import argparse
import functools
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import initialize_model
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bayesopinf_mcmc_check as base
from bayesopinf_trajectory import simulate, REGIMES      # RK4 integrator in JAX

numpyro.set_host_device_count(4)

r = base.r
NQ = r * (r + 1) // 2                 # compact quadratic terms
NS = r * (r - 1) // 2                 # strictly-upper entries of an r x r matrix
IS, JS = np.triu_indices(r, 1)
IQ, JQ = np.triu_indices(r)
SAMPLED = ["none", "epH"]                          # parametrizations that are sampled
STRUCTURES = ["none", "dissA", "epH", "epH+dissA"]  # reported (dissA = truncation)


# ----------------------------------------------------------------------------
# Structured parametrizations
# ----------------------------------------------------------------------------
def skew(v):
    M = jnp.zeros((r, r)).at[IS, JS].set(v)
    return M - M.T


def H_from_S(S):
    """Compact-Kronecker H for q -> sum_k q_k S_k q  (S: r x r x r, each S[k] skew)."""
    cols = S[IQ, :, JQ] + S[JQ, :, IQ]                  # NQ x r: S_i e_j + S_j e_i
    cols = cols * jnp.where(IQ == JQ, 0.5, 1.0)[:, None]
    return cols.T                                       # r x NQ


def sample_O(structure, s):
    c = numpyro.sample("c", dist.Normal(0, s).expand([r]).to_event(1))
    A = numpyro.sample("A", dist.Normal(0, s).expand([r, r]).to_event(2))
    if structure == "epH":
        S = jax.vmap(skew)(numpyro.sample("S", dist.Normal(0, s).expand([r, NS]).to_event(2)))
        H = H_from_S(S)
    else:
        H = numpyro.sample("H", dist.Normal(0, s).expand([r, NQ]).to_event(2))
    return numpyro.deterministic("O", jnp.hstack([c[:, None], A, H]))


def is_dissipative(O_samp):
    A = O_samp[:, :, 1:1 + r]
    return np.linalg.eigvalsh(0.5 * (A + np.swapaxes(A, 1, 2))).max(-1) < 0


def n_free(structure):
    return r + r * r + (r * NS if "epH" in structure else r * NQ)


# ----------------------------------------------------------------------------
# The two likelihoods
# ----------------------------------------------------------------------------
def deriv_model(D, R, sigma, structure, s):
    O = sample_O(structure, s)
    numpyro.sample("r", dist.Normal(O @ D.T, sigma[:, None]).to_event(2), obs=R)


def traj_model(Y, dt, structure, s):
    O = sample_O(structure, s)
    n_traj, K, _ = Y.shape
    s0 = numpyro.sample("s0", dist.Normal(Y[:, 0, :], 1.0).to_event(2))
    sigma_y = numpyro.sample("sigma_y", dist.HalfNormal(0.1))
    Q = jax.vmap(lambda q0: simulate(O, q0, dt, K, 1))(s0)
    Q = jnp.where(jnp.isfinite(Q), Q, 1e6)
    numpyro.sample("y", dist.Normal(Q, sigma_y).to_event(3), obs=Y)


# ----------------------------------------------------------------------------
# MAP + Laplace warm start + NUTS with the Laplace matrix held fixed
# ----------------------------------------------------------------------------
def fit(model, init_values, seed=0, num_warmup=1000, num_samples=1000):
    info = initialize_model(jax.random.PRNGKey(seed), model,
                            init_strategy=init_to_value(values=init_values))
    flat0, unravel = ravel_pytree(info.param_info.z)
    U = lambda x: info.potential_fn(unravel(x))
    vg, hess = jax.jit(jax.value_and_grad(U)), jax.jit(jax.hessian(U))

    def fun(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float)

    t0 = time.time()
    opt = minimize(fun, np.asarray(flat0), jac=True, method="trust-exact",
                   hess=lambda x: np.asarray(hess(jnp.asarray(x)), dtype=float),
                   options=dict(maxiter=500, gtol=1e-6))
    Hs = np.asarray(hess(jnp.asarray(opt.x)))
    w, V = np.linalg.eigh(0.5 * (Hs + Hs.T))
    Hinv = (V / np.clip(w, 1e-8 * w.max(), None)) @ V.T
    sites = set(info.param_info.z)
    map_vals = {k: v for k, v in info.postprocess_fn(unravel(jnp.asarray(opt.x))).items()
                if k in sites}
    t_map = time.time() - t0

    # Laplace-WHITENED NUTS: sample u with theta = theta_MAP + C u, C C^T = H^-1, and let
    # NUTS adapt a dense mass matrix in u. The whitening puts every parameter on an O(1)
    # scale (so NumPyro's mass-matrix regularization is harmless), while the adaptation
    # corrects non-Gaussian geometry the Laplace approximation misses, e.g. a dissipativity
    # constraint that is active, which makes the posterior in L quartic near L = 0.
    C = np.linalg.cholesky(Hinv + 1e-12 * np.eye(len(Hinv)))
    x_map, Cj = jnp.asarray(opt.x), jnp.asarray(C)
    pot_u = lambda u: U(x_map + Cj @ u)
    kernel = NUTS(potential_fn=pot_u, dense_mass=True, target_accept_prob=0.9, max_tree_depth=8)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=4, progress_bar=False)
    u0 = 0.1 * jax.random.normal(jax.random.PRNGKey(seed + 2), (4, len(opt.x)))
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed + 1), init_params=u0, extra_fields=("diverging", "num_steps"))
    u = mcmc.get_samples(group_by_chain=True)                       # chains x draws x n
    to_O = jax.jit(jax.vmap(lambda uu: info.postprocess_fn(unravel(x_map + Cj @ uu))["O"]))
    O_c = np.asarray(to_O(u.reshape(-1, u.shape[-1]))).reshape(*u.shape[:2], r, -1)
    sig = None
    if "sigma_y" in sites:
        to_s = jax.jit(jax.vmap(lambda uu: info.postprocess_fn(unravel(x_map + Cj @ uu))["sigma_y"]))
        sig = np.asarray(to_s(u.reshape(-1, u.shape[-1])))
    t_nuts = time.time() - t0
    ef = mcmc.get_extra_fields()
    flat_O = O_c.reshape(*O_c.shape[:2], -1)
    diag = dict(div=int(np.sum(ef["diverging"])), steps=float(np.mean(ef["num_steps"])),
                ess_min=float(np.nanmin(effective_sample_size(flat_O))),   # nan: entries fixed at 0 by epH
                rhat_max=float(np.nanmax(split_gelman_rubin(flat_O))),
                t_map=t_map, t_nuts=t_nuts, map_iters=opt.nit)
    out = dict(O=O_c.reshape(-1, *O_c.shape[2:]), map=map_vals, diag=diag)
    if sig is not None:
        out["sigma_y"] = sig
    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def score(O_samp, prob, t_pred, n_pf=300, seed=1):
    O_true = prob["O_true"]
    mean = O_samp.mean(0)
    lo, hi = np.percentile(O_samp, [2.5, 97.5], axis=0)
    rng = np.random.default_rng(seed)
    Y = base.pushforward(O_samp[rng.choice(len(O_samp), n_pf, replace=False)], t_pred, prob["q0"])
    truth = solve_ivp(base.rom_rhs(O_true), (t_pred[0], t_pred[-1]), prob["q0"],
                      t_eval=t_pred, rtol=1e-10, atol=1e-12).y
    blo, bhi = np.percentile(Y, [2.5, 97.5], axis=0)
    return dict(rel_err=np.linalg.norm(mean - O_true) / np.linalg.norm(O_true),
                coverage=np.mean((O_true >= lo) & (O_true <= hi)),
                width=np.mean(hi - lo), finite=len(Y) / n_pf,
                traj_cov=np.mean((truth >= blo) & (truth <= bhi)),
                band_w=np.mean(bhi - blo), Y=Y, truth=truth)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=REGIMES, default="scarce")
    ap.add_argument("--prior-scale", type=float, default=1.0)
    ap.add_argument("--horizon", type=float, default=2.0)
    args = ap.parse_args()

    cfg = REGIMES[args.regime]
    prob = base.build_problem(**cfg)
    t = prob["t"]
    dt = t[1] - t[0]
    Y = prob["D"][:, 1:1 + r].reshape(cfg["n_traj"], cfg["K"], r)
    t_pred = np.linspace(0, args.horizon * t[-1], 400)
    D, R = jnp.asarray(prob["D"]), jnp.asarray(prob["R"])
    sigma = jnp.asarray(np.sqrt(prob["sigma2"]))
    s = args.prior_scale
    print(f"regime = {args.regime}: {cfg['n_traj']} trajectories x {cfg['K']} snapshots, "
          f"noise {cfg['noise']}\n")

    zero_init = {
        "none": dict(c=jnp.zeros(r), A=-jnp.eye(r), H=jnp.zeros((r, NQ))),
        "epH":  dict(c=jnp.zeros(r), A=-jnp.eye(r), S=jnp.zeros((r, NS))),
    }

    results = {}
    for st in SAMPLED:
        m_d = functools.partial(deriv_model, D=D, R=R, sigma=sigma, structure=st, s=s)
        fd = fit(m_d, zero_init[st])
        # trajectory likelihood, started from the derivative-based MAP of the same structure
        init_t = dict(fd["map"], s0=jnp.asarray(Y[:, 0, :]), sigma_y=0.01)
        m_t = functools.partial(traj_model, Y=jnp.asarray(Y), dt=dt, structure=st, s=s)
        ft = fit(m_t, init_t)
        for lik, f in [("derivative", fd), ("trajectory", ft)]:
            dg = f["diag"]
            extra = f", sigma_y = {f['sigma_y'].mean():.4f}" if "sigma_y" in f else ""
            print(f"{st + ' / ' + lik:>24s}: {n_free(st):2d} op params | MAP {dg['map_iters']:3d} it "
                  f"{dg['t_map']:5.1f}s | NUTS {dg['t_nuts']:5.0f}s, {dg['steps']:4.0f} steps/it, "
                  f"div {dg['div']}, minESS {dg['ess_min']:.0f}, R-hat {dg['rhat_max']:.3f}{extra}")
            keep = is_dissipative(f["O"])
            variants = [(st, f["O"], 1.0),
                        ("dissA" if st == "none" else "epH+dissA", f["O"][keep], keep.mean())]
            for name, O_s, acc in variants:
                if len(O_s) < 300:
                    print(f"   {name} / {lik}: only {len(O_s)} dissipative draws, skipped")
                    continue
                results[f"{name} / {lik}"] = dict(O=O_s, accept=acc, **score(O_s, prob, t_pred))

    hdr = (f"\n{'structure / likelihood':>24s} | {'accept':>6s} | {'op err':>6s} | {'op cov':>6s} | "
           f"{'CI width':>8s} | {'stable':>6s} | {'traj cov':>8s} | {'band width':>10s}")
    print(hdr); print("-" * len(hdr))
    for k, v in results.items():
        print(f"{k:>24s} | {v['accept']:6.2f} | {v['rel_err']:6.3f} | {v['coverage']:6.2f} | "
              f"{v['width']:8.3f} | {v['finite']:6.2f} | {v['traj_cov']:8.2f} | {v['band_w']:10.4f}")

    # --- forest plot ---------------------------------------------------------
    O_true = prob["O_true"]
    d = O_true.shape[1]
    names = ["c"] + [f"A{k}" for k in range(r)] + [f"H{k}" for k in range(d - 1 - r)]
    P = len(results)
    fig, axes = plt.subplots(r, 1, figsize=(12, 3 * r), sharex=True)
    for i, ax in enumerate(axes):
        for k, (key, v) in enumerate(results.items()):
            x = np.arange(d) + (k - (P - 1) / 2) * 0.12
            m = v["O"][:, i].mean(0)
            lo, hi = np.percentile(v["O"][:, i], [2.5, 97.5], axis=0)
            ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o" if "deriv" in key else "s",
                        ms=3, lw=1.2, color=f"C{STRUCTURES.index(key.split(' /')[0])}",
                        alpha=0.55 if "deriv" in key else 1.0, label=key if i == 0 else None)
        ax.plot(np.arange(d), O_true[i], "kx", ms=8, mew=2, label="truth" if i == 0 else None)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_ylabel(f"row {i}")
    axes[-1].set_xticks(np.arange(d), names)
    axes[0].legend(ncol=4, fontsize=7, loc="upper right")
    fig.suptitle(f"Operator posteriors (95% CI), {args.regime} data  "
                 f"(circles: derivative, squares: trajectory)")
    fig.tight_layout()
    fig.savefig(f"structured_operators_{args.regime}.png", dpi=300)
    fig.savefig(f"structured_operators_{args.regime}.pdf")

    # --- prediction bands ----------------------------------------------------
    fig, axes = plt.subplots(r, P, figsize=(2.9 * P, 2.3 * r), sharex=True, sharey="row")
    for k, (key, v) in enumerate(results.items()):
        col = f"C{STRUCTURES.index(key.split(' /')[0])}"
        lo, med, hi = np.percentile(v["Y"], [2.5, 50, 97.5], axis=0)
        for j in range(r):
            ax = axes[j, k]
            ax.fill_between(t_pred, lo[j], hi[j], color=col, alpha=0.3)
            ax.plot(t_pred, med[j], color=col, lw=1)
            ax.plot(t_pred, v["truth"][j], "k--", lw=1)
            ax.axvline(t[-1], color="0.5", lw=0.8, ls=":")
            if j == 0:
                ax.set_title(f"{key}\n(stable {v['finite']:.0%})", fontsize=8)
            if k == 0:
                ax.set_ylabel(f"$\\hat q_{j}$")
    for ax in axes[-1]:
        ax.set_xlabel("t")
    fig.suptitle("95% prediction bands (dashed: truth, dotted: end of training data)")
    fig.tight_layout()
    fig.savefig(f"structured_pushforward_{args.regime}.png", dpi=300)
    fig.savefig(f"structured_pushforward_{args.regime}.pdf")
    print(f"\nsaved structured_operators_{args.regime}.{{png,pdf}}, "
          f"structured_pushforward_{args.regime}.{{png,pdf}}")


if __name__ == "__main__":
    main()
