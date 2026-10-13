"""
Toy quadratic systems with KNOWN operators, used to generate data for Bayesian OpInf tests.

Every system has the OpInf form (state dimension r = 3)

    dq/dt = c + A q + H (q ⊗ q)_compact,      (q ⊗ q)_compact = [q_i q_j for i <= j]

so the data-generating model is exactly in the model class and the true operators are known.

Systems (select by name in true_operators):

    damped        damping + rotation, energy-preserving H. Every trajectory spirals into a stable
                  fixed point within a few time units.
    weak_damping  same rotation and H, damping 0.05 on every state and c = 0. Slowly decaying
                  oscillations (decay time ~20), so a test window after training still oscillates.
                  (c = 0 matters: with the original c the fixed point moves away from the origin,
                  and linearized there the quadratic term adds strong damping of its own.)
    limit_cycle   mean-field model of cylinder vortex shedding (Noack et al. 2003, J. Fluid Mech.
                  497, 335-363) with relaxation rate 1:
                      x' = s x - w y - x z,   y' = w x + s y - y z,   z' = -z + x^2 + y^2
                  Linearly unstable (A is NOT dissipative), saturated by an energy-preserving H
                  -> stable limit cycle x^2 + y^2 = z = s (limit cycle of radius sqrt(s)),
                    period 2 pi / w. Sustained oscillation.

To add a system: write a function returning (c, A, H) and register it in SYSTEMS.

Run `python toy_systems.py [--T 12]` to print each system's structure and plot sample trajectories
over [0, T].
"""

import numpy as np

r = 3                                              # state dimension of every toy system
IU, JU = np.triu_indices(r)                        # compact-Kronecker index pairs (i <= j)
COL = {(int(i), int(j)): m for m, (i, j) in enumerate(zip(IU, JU))}   # (i, j) -> column of H



# Shared helpers
def compact_kron(q):
    """Non-redundant quadratic terms q_i q_j, i <= j. Works on (r,) or (r, K)."""
    i, j = np.triu_indices(q.shape[0])
    return q[i] * q[j]


def operators_to_matrix(c, A, H):
    """Stack into O = [c, A, H] (r x d); row i of O is o_i."""
    return np.hstack([c[:, None], A, H])


def rom_rhs(O):
    """Right-hand side f(t, q) = c + A q + H (q ⊗ q) for solve_ivp."""
    c, A, H = O[:, 0], O[:, 1:1 + r], O[:, 1 + r:]
    return lambda t, q: c + A @ q + H @ compact_kron(q)


def H_from_skew(S):
    """Compact H for the energy-preserving quadratic q -> sum_k q_k S_k q (each S_k skew)."""
    H = np.zeros((r, len(IU)))
    for m, (i, j) in enumerate(zip(IU, JU)):
        v = S[i][:, j] + S[j][:, i]                    # S_i e_j + S_j e_i
        if i == j:
            H[:, m] = v / 2                 # halved on the diagonal (q_i^2 counted once)
        else:
            H[:, m] = v       
    return H



# Systems
_S_ORIGINAL = [np.array([[0, a, b], [-a, 0, e], [-b, -e, 0]], dtype=float)
               for a, b, e in [(0.6, -0.3, 0.2), (-0.4, 0.5, 0.3), (0.2, 0.1, -0.5)]]
_ROTATION = np.array([[0, 1.0, 0], [-1.0, 0, 0.5], [0, -0.5, 0]])


def _damped_rotation(damping, c=(0.5, 0.0, -0.3)):
    A = -np.diag(damping) + _ROTATION
    return np.array(c, dtype=float), A, H_from_skew(_S_ORIGINAL)


def original():
    """Damping + rotation, energy-preserving H: decays to a stable fixed point."""
    return _damped_rotation([0.4, 0.8, 1.2])


def weak_damping():
    """Same rotation and H, weak damping (0.05) and c = 0: slowly decaying oscillations."""
    return _damped_rotation([0.05, 0.05, 0.05], c=(0.0, 0.0, 0.0))


# def limit_cycle(s=0.5, w=2.0):
def limit_cycle(s=0.1, w=1.0):
# def limit_cycle(s=-1, w=1.0):
    """Mean-field vortex-shedding model: stable limit cycle of radius sqrt(s), period 2 pi / w."""
    c = np.zeros(r)
    A = np.array([[s, -w, 0.0], [w, s, 0.0], [0.0, 0.0, -1.0]])
    H = np.zeros((r, len(IU)))
    H[0, COL[(0, 2)]] = -1.0                           # x' : - x z
    H[1, COL[(1, 2)]] = -1.0                           # y' : - y z
    H[2, COL[(0, 0)]] = 1.0                            # z' : + x^2
    H[2, COL[(1, 1)]] = 1.0                            # z' : + y^2
    return c, A, H


SYSTEMS = {"damped": original, "weak_damping": weak_damping, "limit_cycle": limit_cycle}


def system_tag(system):
    """File-name tag of a system: its name plus the default values of its parameters, e.g.
    limit_cycle_s-0.1_w1. Read from the current defaults, so editing them changes the file names."""
    import inspect
    params = inspect.signature(SYSTEMS[system]).parameters.values()
    return system + "".join(f"_{q.name}{q.default:g}" for q in params if q.default is not inspect.Parameter.empty)


def true_operators(system="damped"):
    """(c, A, H) of the named system."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; choose from {list(SYSTEMS)}")
    return SYSTEMS[system]()



# Quick look: structure and sample trajectories of every system
if __name__ == "__main__":
    import os
    import argparse
    from scipy.integrate import solve_ivp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # LaTeX text rendering (Computer Modern); needs a LaTeX installation
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"]})

    ap = argparse.ArgumentParser()
    ap.add_argument("--T_train", type=float, default=6.0, help="training time [0, T_train]")
    ap.add_argument("--T_test", type=float, default=12.0, help="total simulation time [0, T_test]")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    
    # random initial conditions for each system, 4 per system
    q0s = rng.uniform(-1.5, 1.5, (1, r))
    t = np.linspace(0, args.T_test, int(round(100 * args.T_test)) + 1)   # 100 samples per time unit
    # rows = states, columns = systems
    fig, axes = plt.subplots(r, len(SYSTEMS), figsize=(4 * len(SYSTEMS), 2.4 * r), sharex=True, squeeze=False)
    for col, name in enumerate(SYSTEMS):
        c, A, H = true_operators(name)
        O = operators_to_matrix(c, A, H)
        qs = rng.standard_normal((200, r))
        
        # Quadratic energy-preserving: q^T H (q kron q) = 0 for all q if H is energy-preserving
        # Check: max |q^T H (q kron q)| over random q
        energy_preserving = max(abs(q @ (H @ compact_kron(q))) for q in qs)
        
        print(f"{name:>13s}: max eig sym(A) = {np.linalg.eigvalsh(0.5 * (A + A.T)).max():+.3f} "
              f"({'dissipative' if np.linalg.eigvalsh(0.5 * (A + A.T)).max() < 0 else 'NOT dissipative'}), "
              f"energy-preserving H: {energy_preserving < 1e-12}")

        # the operators of dq/dt = c + A q + H (q kron q)_compact, one row per state
        hcols = [f"x{i}x{j}" for i, j in zip(IU, JU)]
        print(f"{'':>6s}{'c':>8s} |" + "".join(f"{f'A[:, {j}]':>8s}" for j in range(r)) + " |"
              + "".join(f"{h:>8s}" for h in hcols))
        for i in range(r):
            print(f"{f'dx{i}/dt':>6s}{c[i]:8.3g} |" + "".join(f"{v:8.3g}" for v in A[i]) + " |"
                  + "".join(f"{v:8.3g}" for v in H[i]))
        print()

        for q0 in q0s:
            y = solve_ivp(rom_rhs(O), (0, t[-1]), q0, t_eval=t, rtol=1e-10, atol=1e-12).y
            for k in range(r):
                axes[k, col].plot(t, y[k], lw=1)
        for k in range(r):
            axes[k, col].axvline(args.T_train, color="0.5", ls=":", lw=0.8)
        axes[0, col].set_title(name.replace("_", " "), fontsize=15)   # a bare "_" is an error in LaTeX
    for k in range(r):
        axes[k, 0].set_ylabel(f"$x_{k}$", fontsize=15)
        for j in range(len(SYSTEMS)):
            axes[k, j].tick_params(axis="y", labelsize=15)
        
    
    for a in axes[-1]:
        a.set_xlabel("time [s]", fontsize=15)
        a.tick_params(axis="x", labelsize=15)
    fig.tight_layout()
    # figures go to pdfs/<this file>_pdfs/, like the other scripts
    here, stem = os.path.split(os.path.splitext(os.path.abspath(__file__))[0])
    out_dir = os.path.join(here, "pdfs", stem + "_pdfs")
    os.makedirs(out_dir, exist_ok=True)
    # limit-cycle parameters and T in the name, so runs with other settings do not overwrite each other
    name = f"toy_systems_trajectories_{system_tag('limit_cycle')}_T{args.T_test:g}.pdf"
    fig.savefig(os.path.join(out_dir, name))
    print(f"saved {name} in {out_dir}")
