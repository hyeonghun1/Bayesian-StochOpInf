"""
Bayesian OpInf with non-conjugate priors, sampled with NUTS.

Reuses the toy problem from bayesopinf_mcmc_check.py (same folder), so the true
operators are known and each prior can be scored against them.

Priors per row o_i (entries grouped: g=0 -> [c, A], g=1 -> H):
    tikhonov   o ~ N(0, sigma^2 Gamma^-1), Gamma fixed        (Guo et al. baseline)
    hier_gauss o_j ~ N(0, tau_g^2),         tau_g ~ HalfCauchy (regularization learned;
               o integrated out analytically, NUTS on tau only, o drawn exactly)
    laplace    o_j ~ Laplace(0, tau_g),     tau_g ~ HalfCauchy (Bayesian lasso)
    student_t  o_j ~ tau_g * t_3,           tau_g ~ HalfCauchy (heavy tails)
    horseshoe  regularized horseshoe (Piironen & Vehtari 2017), per group

sigma_i: fixed plug-in (as in Guo et al.) or sampled (--learn-sigma).

Data: build_problem() from bayesopinf_mcmc_check.py, with the data-generating system from
toy_systems.py (--system) and derivatives from local polynomial smoothing ("lpr", default) or
plain finite differences ("fd") (--deriv).

Note: with a single trajectory D is nearly rank-deficient, and smoothed derivatives make the plug-in
sigma small. Guo's prior is scaled by sigma^2, so tikhonov then shrinks the unidentified directions
very hard (narrow, overconfident intervals), while the learned-scale priors (tau ~ HalfCauchy(1)) leave
them almost free (wide intervals, many unstable rollouts). Compare --regime rich.

Usage:
    python bayesopinf_priors.py                       # scarce data, all priors
    python bayesopinf_priors.py --regime rich
    python bayesopinf_priors.py --priors tikhonov horseshoe --learn-sigma
    python bayesopinf_priors.py --system limit_cycle --deriv fd
Figures go to pdfs/bayesopinf_priors_pdfs/.
"""

import argparse
import os
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bayesopinf_mcmc_check as base     # build_problem, rom_rhs, pushforward, r

numpyro.set_host_device_count(4)

REGIMES = {
    # few, noisy data: the prior matters
    "scarce": dict(noise=0.01, T_train=4.0, K=300, n_traj=1),
    # the setting of the sanity check: data dominate, priors should roughly agree
    "rich":   dict(noise=0.01, T_train=4.0, K=300, n_traj=5),
}
ALL_PRIORS = ["tikhonov", "hier_gauss", "laplace", "student_t", "horseshoe"]


# ----------------------------------------------------------------------------
# Prior library. Horseshoe is non-centered (o = scale * z): its per-coefficient
# local scales create funnels. Group-scale priors are centered by default.
# ----------------------------------------------------------------------------
N_GROUPS = 2
NONCENTERED = False


def sample_operators(prior, d, groups, Gamma, sigma):
    n_g = N_GROUPS

    if prior == "tikhonov":
        z = numpyro.sample("z", dist.Normal(0, 1).expand([d]).to_event(1))
        return numpyro.deterministic("o", sigma * z / jnp.sqrt(jnp.diag(Gamma)))

    tau = numpyro.sample("tau", dist.HalfCauchy(1.0).expand([n_g]).to_event(1))
    scale = tau[groups]

    # Group-scale priors are written CENTERED (o ~ p(. | tau) directly). Each tau
    # is shared by 4-6 coefficients that the data largely identify, which is the
    # regime where centered sampling works and non-centered gives a curved
    # (tau, z) ridge. Switch with --noncentered to see the difference.
    if prior in ("hier_gauss", "laplace", "student_t") and not NONCENTERED:
        base_d = {"hier_gauss": dist.Normal(0, scale),
                  "laplace": dist.Laplace(0, scale),
                  "student_t": dist.StudentT(3.0, 0, scale)}[prior]
        return numpyro.sample("o", base_d.to_event(1))

    if prior == "hier_gauss":
        z = numpyro.sample("z", dist.Normal(0, 1).expand([d]).to_event(1))
    elif prior == "laplace":
        z = numpyro.sample("z", dist.Laplace(0, 1).expand([d]).to_event(1))
    elif prior == "student_t":
        z = numpyro.sample("z", dist.StudentT(3.0, 0, 1).expand([d]).to_event(1))
    elif prior == "horseshoe":
        # regularized horseshoe: local lam_j, global tau_g, slab width c
        lam = numpyro.sample("lam", dist.HalfCauchy(1.0).expand([d]).to_event(1))
        c2 = numpyro.sample("c2", dist.InverseGamma(2.0, 2.0 * 2.0**2))   # slab ~ scale 2
        lam_t = jnp.sqrt(c2 * lam**2 / (c2 + scale**2 * lam**2))
        z = numpyro.sample("z", dist.Normal(0, 1).expand([d]).to_event(1))
        scale = scale * lam_t
    else:
        raise ValueError(prior)
    return numpyro.deterministic("o", scale * z)


# ----------------------------------------------------------------------------
# Collapsed hierarchical Gaussian: o | tau, sigma is conjugate, so integrate o
# out analytically, run NUTS on (tau, sigma) only, then draw o | tau, sigma
# from the closed form. This is the Guo et al. posterior used as an exact block
# inside a hierarchical model, and it removes the o-tau funnel entirely.
#   r ~ N(0, sigma^2 I + D Lam D^T),  Lam = diag(tau_g^2)
#   P = Lam^-1 + D^T D / sigma^2   (all d x d via Woodbury / det. lemma)
# ----------------------------------------------------------------------------
def collapsed_model(DtD, Dtr, rtr, K, groups, sigma_fixed, sigma_scale, learn_sigma):
    tau = numpyro.sample("tau", dist.HalfCauchy(1.0).expand([N_GROUPS]).to_event(1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_scale)) if learn_sigma else sigma_fixed
    s2 = sigma**2
    lam = tau[groups] ** 2
    P = jnp.diag(1.0 / lam) + DtD / s2
    L = jnp.linalg.cholesky(P)
    w = jax.scipy.linalg.solve_triangular(L, Dtr / s2, lower=True)
    logdet = K * jnp.log(s2) + jnp.sum(jnp.log(lam)) + 2 * jnp.sum(jnp.log(jnp.diag(L)))
    quad = rtr / s2 - w @ w
    numpyro.factor("marginal_lik", -0.5 * (quad + logdet + K * jnp.log(2 * jnp.pi)))


def draw_o_given_hyper(DtD, Dtr, groups, tau, sigma, rng):
    """Exact conjugate draws o ~ N(P^-1 D^T r / s2, P^-1), one per hyper-sample."""
    out = np.empty((len(tau), DtD.shape[0]))
    for n in range(len(tau)):
        s2 = sigma[n] ** 2
        P = np.diag(1.0 / tau[n][groups] ** 2) + DtD / s2
        L = np.linalg.cholesky(P)
        mean = np.linalg.solve(P, Dtr / s2)
        out[n] = mean + np.linalg.solve(L.T, rng.standard_normal(len(mean)))
    return out


def make_row_model(prior, learn_sigma):
    # prior / learn_sigma are Python-level switches, so close over them
    # instead of passing them through JAX tracing
    def row_model(D, r_i, groups, Gamma, sigma_fixed, sigma_scale):
        return _row_model(D, r_i, groups, Gamma, prior,
                          None if learn_sigma else sigma_fixed, sigma_scale)
    return row_model


def _row_model(D, r_i, groups, Gamma, prior, sigma_fixed, sigma_scale):
    if sigma_fixed is None:
        sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_scale))
    else:
        sigma = sigma_fixed
    o = sample_operators(prior, D.shape[1], groups, Gamma, sigma)
    numpyro.sample("r", dist.Normal(D @ o, sigma), obs=r_i)


def run_prior(prob, prior, learn_sigma, seed=0, num_warmup=1500, num_samples=2000):
    D, R = jnp.asarray(prob["D"]), prob["R"]
    d = D.shape[1]
    groups = jnp.asarray([0] * (1 + base.r) + [1] * (d - 1 - base.r))
    rows, stats = [], dict(div=0, ess_min=np.inf, rhat_max=0.0)
    t0 = time.time()
    for i in range(R.shape[0]):
        if prior == "hier_gauss":
            Dn, rn = prob["D"], R[i]
            DtD, Dtr = Dn.T @ Dn, Dn.T @ rn
            kernel = NUTS(collapsed_model, dense_mass=True, target_accept_prob=0.95)
            mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                        num_chains=4, progress_bar=False)
            sig_fix = float(np.sqrt(prob["sigma2"][i]))
            mcmc.run(jax.random.PRNGKey(seed + 10 * i),
                     DtD=jnp.asarray(DtD), Dtr=jnp.asarray(Dtr), rtr=float(rn @ rn),
                     K=Dn.shape[0], groups=groups, sigma_fixed=sig_fix,
                     sigma_scale=float(np.std(rn)), learn_sigma=learn_sigma,
                     extra_fields=("diverging",))
            hs = mcmc.get_samples(group_by_chain=True)
            tau = np.asarray(hs["tau"]).reshape(-1, N_GROUPS)
            sig = (np.asarray(hs["sigma"]).reshape(-1) if learn_sigma
                   else np.full(len(tau), sig_fix))
            o = draw_o_given_hyper(DtD, Dtr, np.asarray(groups), tau, sig,
                                   np.random.default_rng(seed + i))
            o = o.reshape(4, num_samples, d)
        else:
            kernel = NUTS(make_row_model(prior, learn_sigma), dense_mass=True,
                          target_accept_prob=0.99 if prior == "horseshoe" else 0.95)
            mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                        num_chains=4, progress_bar=False)
            mcmc.run(jax.random.PRNGKey(seed + 10 * i),
                     D=D, r_i=jnp.asarray(R[i]), groups=groups,
                     Gamma=jnp.asarray(prob["Gamma"]),
                     sigma_fixed=float(np.sqrt(prob["sigma2"][i])),
                     sigma_scale=float(np.std(R[i])),
                     extra_fields=("diverging",))
            o = np.asarray(mcmc.get_samples(group_by_chain=True)["o"])   # chains x draws x d
        stats["div"] += int(np.sum(mcmc.get_extra_fields()["diverging"]))
        stats["ess_min"] = min(stats["ess_min"], float(np.min(effective_sample_size(o))))
        stats["rhat_max"] = max(stats["rhat_max"], float(np.max(split_gelman_rubin(o))))
        rows.append(o.reshape(-1, d))
    stats["time"] = time.time() - t0
    return np.stack(rows, axis=1), stats                     # draws x r x d


# ----------------------------------------------------------------------------
# Scoring against the truth
# ----------------------------------------------------------------------------
def score_operators(O_samp, O_true):
    mean = O_samp.mean(0)
    lo, hi = np.percentile(O_samp, [2.5, 97.5], axis=0)
    return dict(
        rel_err=np.linalg.norm(mean - O_true) / np.linalg.norm(O_true),
        coverage=np.mean((O_true >= lo) & (O_true <= hi)),
        width=np.mean(hi - lo),
        # entries that are exactly zero in the truth: does the posterior shrink them?
        zero_abs=np.mean(np.abs(mean[O_true == 0])) if np.any(O_true == 0) else np.nan,
    )


def score_pushforward(O_samp, prob, t_pred, n=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(O_samp), n, replace=False)
    Y = base.pushforward(O_samp[idx], t_pred, prob["q0"])
    finite = len(Y) / n
    truth = solve_ivp(base.rom_rhs(prob["O_true"]), (t_pred[0], t_pred[-1]), prob["q0"],
                      t_eval=t_pred, rtol=1e-10, atol=1e-12).y
    lo, hi = np.percentile(Y, [2.5, 97.5], axis=0)
    return Y, truth, dict(finite=finite,
                          traj_cov=np.mean((truth >= lo) & (truth <= hi)),
                          band_w=np.mean(hi - lo))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=REGIMES, default="scarce")
    ap.add_argument("--priors", nargs="+", choices=ALL_PRIORS, default=ALL_PRIORS)
    ap.add_argument("--learn-sigma", action="store_true")
    ap.add_argument("--noncentered", action="store_true",
                    help="non-centered parametrization for hier_gauss/laplace/student_t")
    ap.add_argument("--horizon", type=float, default=2.0,
                    help="predict to horizon * training T (extrapolation in time)")
    ap.add_argument("--system", default="original", choices=list(base.SYSTEMS),
                    help="true ROM generating the data (defined in toy_systems.py)")
    ap.add_argument("--deriv", default="lpr", choices=["lpr", "fd"],
                    help="derivative estimates: local polynomial smoothing or finite differences")
    args = ap.parse_args()
    global NONCENTERED
    NONCENTERED = args.noncentered

    prob = base.build_problem(**REGIMES[args.regime], system=args.system, deriv=args.deriv)
    O_true = prob["O_true"]
    T_train = prob["t"][-1]
    t_pred = np.linspace(0, args.horizon * T_train, 400)
    print(f"system = {args.system}, regime = {args.regime}, deriv = {args.deriv}, "
          f"learn_sigma = {args.learn_sigma}, "
          f"prediction horizon = {t_pred[-1]:.1f} (training ends at {T_train:.1f})\n")

    results = {}
    for prior in args.priors:
        O_samp, st = run_prior(prob, prior, args.learn_sigma)
        sc = score_operators(O_samp, O_true)
        Y, truth, pf = score_pushforward(O_samp, prob, t_pred)
        results[prior] = dict(O=O_samp, Y=Y, truth=truth, **st, **sc, **pf)
        print(f"{prior:>10s} | div {st['div']:4d} | minESS {st['ess_min']:6.0f} | "
              f"R-hat {st['rhat_max']:.3f} | {st['time']:5.1f}s")

    hdr = (f"\n{'prior':>10s} | {'op rel err':>10s} | {'op 95% cov':>10s} | {'op CI width':>11s} | "
           f"{'|mean| at true 0':>16s} | {'stable':>6s} | {'traj 95% cov':>12s} | {'band width':>10s}")
    print(hdr); print("-" * len(hdr))
    for p, s in results.items():
        print(f"{p:>10s} | {s['rel_err']:10.3f} | {s['coverage']:10.2f} | {s['width']:11.3f} | "
              f"{s['zero_abs']:16.3f} | {s['finite']:6.2f} | {s['traj_cov']:12.2f} | {s['band_w']:10.3f}")

    tag = (("" if args.system == "original" else f"{args.system}_") + args.regime
           + ("" if args.deriv == "lpr" else f"_{args.deriv}")
           + ("_learnsigma" if args.learn_sigma else ""))
    # figures go to pdfs/<this file>_pdfs/, like bayesopinf_mcmc_check.py
    here, stem = os.path.split(os.path.splitext(os.path.abspath(__file__))[0])
    out_dir = os.path.join(here, "pdfs", stem + "_pdfs")
    os.makedirs(out_dir, exist_ok=True)
    out = lambda name: os.path.join(out_dir, f"{name}_{tag}.pdf")

    # --- forest plot: posterior 95% CI of every operator entry, by prior ------
    d = O_true.shape[1]
    names = ["c"] + [f"A{k}" for k in range(base.r)] + [f"H{k}" for k in range(d - 1 - base.r)]
    P = len(results)
    fig, axes = plt.subplots(O_true.shape[0], 1, figsize=(11, 2.8 * O_true.shape[0]), sharex=True)
    for i, ax in enumerate(axes):
        for k, (p, s) in enumerate(results.items()):
            x = np.arange(d) + (k - (P - 1) / 2) * 0.14
            m = s["O"][:, i].mean(0)
            lo, hi = np.percentile(s["O"][:, i], [2.5, 97.5], axis=0)
            ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o", ms=3, lw=1.2,
                        color=f"C{k}", label=p if i == 0 else None)
        ax.plot(np.arange(d), O_true[i], "kx", ms=8, mew=2, label="truth" if i == 0 else None)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_ylabel(f"row {i}")
    axes[-1].set_xticks(np.arange(d), names)
    axes[0].legend(ncol=P + 1, fontsize=8, loc="upper right")
    fig.suptitle(f"Operator posteriors (95% CI), {args.system}, {args.regime} data")
    fig.tight_layout()
    # fig.savefig(f"priors_operators_{tag}.png", dpi=300)
    fig.savefig(out("priors_operators"))

    # --- push-forward: prediction bands per prior vs truth -------------------
    fig, axes = plt.subplots(base.r, P, figsize=(3.2 * P, 2.3 * base.r), sharex=True, sharey="row",
                             squeeze=False)
    for k, (p, s) in enumerate(results.items()):
        lo, med, hi = np.percentile(s["Y"], [2.5, 50, 97.5], axis=0)
        for j in range(base.r):
            ax = axes[j, k]
            ax.fill_between(t_pred, lo[j], hi[j], color=f"C{k}", alpha=0.3)
            ax.plot(t_pred, med[j], color=f"C{k}", lw=1)
            ax.plot(t_pred, s["truth"][j], "k--", lw=1)
            ax.axvline(T_train, color="0.5", lw=0.8, ls=":")
            if j == 0:
                ax.set_title(f"{p}  (stable {s['finite']:.0%})", fontsize=9)
            if k == 0:
                ax.set_ylabel(f"$\\hat q_{j}$")
    for ax in axes[-1]:
        ax.set_xlabel("t")
    fig.suptitle("95% prediction bands (dashed: truth, dotted: end of training data)")
    fig.tight_layout()
    # fig.savefig(f"priors_pushforward_{tag}.png", dpi=300)
    fig.savefig(out("priors_pushforward"))
    print(f"\nsaved priors_operators_{tag}.pdf, priors_pushforward_{tag}.pdf in {out_dir}")


if __name__ == "__main__":
    main()
