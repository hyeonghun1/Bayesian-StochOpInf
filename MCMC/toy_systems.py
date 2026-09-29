"""
Toy quadratic systems with KNOWN operators, used to generate data for Bayesian OpInf tests.

Every system has the OpInf form (state dimension r = 3)

    dq/dt = c + A q + H (q ⊗ q)_compact,      (q ⊗ q)_compact = [q_i q_j for i <= j]

so the data-generating model is exactly in the model class and the true operators are known.

Systems (select by name in true_operators):

    original      damping + rotation, energy-preserving H. Every trajectory spirals into a stable
                  fixed point within a few time units.
    weak_damping  same rotation and H, damping 0.05 on every state and c = 0. Slowly decaying
                  oscillations (decay time ~20), so a test window after training still oscillates.
                  (c = 0 matters: with the original c the fixed point moves away from the origin,
                  and linearized there the quadratic term adds strong damping of its own.)
    limit_cycle   mean-field model of cylinder vortex shedding (Noack et al. 2003, J. Fluid Mech.
                  497, 335-363) with relaxation rate 1:
                      x' = s x - w y - x z,   y' = w x + s y - y z,   z' = -z + x^2 + y^2
                  Linearly unstable (A is NOT dissipative), saturated by an energy-preserving H
                  -> stable limit cycle x^2 + y^2 = z = s, period 2 pi / w. Sustained oscillation.

To add a system: write a function returning (c, A, H) and register it in SYSTEMS.

Run `python toy_systems.py` to print each system's structure and plot sample trajectories.
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
        H[:, m] = v / 2 if i == j else v               # halved on the diagonal (q_i^2 counted once)
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


def limit_cycle(s=0.5, w=2.0):
    """Mean-field vortex-shedding model: stable limit cycle of radius sqrt(s), period 2 pi / w."""
    c = np.zeros(r)
    A = np.array([[s, -w, 0.0], [w, s, 0.0], [0.0, 0.0, -1.0]])
    H = np.zeros((r, len(IU)))
    H[0, COL[(0, 2)]] = -1.0                           # x' : - x z
    H[1, COL[(1, 2)]] = -1.0                           # y' : - y z
    H[2, COL[(0, 0)]] = 1.0                            # z' : + x^2
    H[2, COL[(1, 1)]] = 1.0                            # z' : + y^2
    return c, A, H


SYSTEMS = {"original": original, "weak_damping": weak_damping, "limit_cycle": limit_cycle}


def true_operators(system="original"):
    """(c, A, H) of the named system."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; choose from {list(SYSTEMS)}")
    return SYSTEMS[system]()


# ----------------------------------------------------------------------------
# Quick look: structure and sample trajectories of every system
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from scipy.integrate import solve_ivp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    q0s = rng.uniform(-1.5, 1.5, (4, r))               # same kind of ICs as build_problem
    t = np.linspace(0, 12, 1201)
    fig, axes = plt.subplots(len(SYSTEMS), r, figsize=(12, 2.4 * len(SYSTEMS)), sharex=True)
    for row, name in enumerate(SYSTEMS):
        c, A, H = true_operators(name)
        O = operators_to_matrix(c, A, H)
        qs = rng.standard_normal((200, r))
        ep = max(abs(q @ (H @ compact_kron(q))) for q in qs)
        print(f"{name:>13s}: max eig sym(A) = {np.linalg.eigvalsh(0.5 * (A + A.T)).max():+.3f} "
              f"({'dissipative' if np.linalg.eigvalsh(0.5 * (A + A.T)).max() < 0 else 'NOT dissipative'}), "
              f"energy-preserving H: {ep < 1e-12}")
        for q0 in q0s:
            y = solve_ivp(rom_rhs(O), (0, t[-1]), q0, t_eval=t, rtol=1e-10, atol=1e-12).y
            for k in range(r):
                axes[row, k].plot(t, y[k], lw=1)
        for k in range(r):
            axes[row, k].axvline(6.0, color="0.5", ls=":", lw=0.8)
            axes[row, k].set_title(f"{name}: q{k}", fontsize=9)
    for a in axes[-1]:
        a.set_xlabel("t  (dotted: default end of training)")
    fig.tight_layout()
    fig.savefig("toy_systems_trajectories.pdf")
    print("saved toy_systems_trajectories.pdf")
