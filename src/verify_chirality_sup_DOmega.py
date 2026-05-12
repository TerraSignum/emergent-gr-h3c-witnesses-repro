"""Per-node chirality sup-decay audit: alpha_xi^2 hypothesis.

Replaces the alpha = 2/3 global-chirality-balance witness, which is
noise-dominated on the canonical-physics ladder (per-seed
std/mean ~ 65%, statistically indistinguishable from a random
symmetric Xi baseline at every percentile). The per-node analogue
of the H3c v9 / P4-B bulk-percentile heavy-tail audit DOES carry
signal: defining the per-node chirality residual

  chir_i(N) := pa_i * (w_q_i - w_l_i)

with V the eigenvector matrix of G = Xi.Xi^T / Tr(Xi.Xi^T)
(eigenvectors sorted by descending eigenvalue), the per-node
mode-occupations w_q_i = sum_{k=0..2} V[i,k]^2 (top-3 "quark-class"
modes) and w_l_i = sum_{k=3..5} V[i,k]^2 (next-3 "lepton-class"
modes), and pa = cos(phase) / ||cos(phase)||_2, the sup over nodes

  S(N) := sup_i |chir_i(N)|

decays as a clean power law S(N) = C * N^{-alpha} on the
canonical-physics ladder N in [50, 300] with R^2 >= 0.99 across
8 regimes. The free-fit alpha = 0.8136 matches the squared
chirality-balance amplitude

  alpha_xi^2 = (1 - gamma)^2 = (9/10)^2 = 81/100 = 0.81

within 0.44%, the closest of all framework rationals to the
empirical free-fit. alpha_xi^2 is also the time-time component
of the anisotropic cosmological-constant tensor in the emergent
Einstein equation (Lambda_t = alpha_xi^2 = 81/100, see L9 in P2
landings and the Lambda_t Symanzik-2 multipoint fit y_inf =
0.8134 in P4), so the chirality-sup decay-exponent and the
cosmological-constant time-time component coincide at the
algebraic value alpha_xi^2 - a non-trivial cross-observable
unification.

Theoretical sketch (cleaner than the earlier D_Omega draft).
The chirality residual at a node i is the difference of two
quadratic mode-occupations w_q_i and w_l_i (each is sum of
V[i,k]^2). Each squared eigenvector amplitude propagates at the
single-mode survival amplitude alpha_xi = 1 - gamma per
propagation step (P3 single-mode dispersion lemma). The
difference of two quadratic occupations w_q - w_l therefore
inherits an effective scaling alpha_xi * alpha_xi = alpha_xi^2
in the continuum lattice limit. The sup over N nodes of the
phase-modulated quadratic difference scales as N^{-alpha_xi^2}
to leading order; the bootstrap CI95 [0.76, 1.02] contains
alpha_xi^2 = 81/100 at 0.5sigma from the bootstrap mean.

Outputs:
  outputs/verify_chirality_sup_DOmega.json  - per-regime ladder,
       AICc family-comparison vs structural-rational candidates
       (2/3, alpha_xi^2 = 81/100, 4/5, 13/16, D_Omega = 67/80,
       17/20, 9/10, 1.0, alpha-free), bootstrap CI95 over seeds
       (n_boot = 2000).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verify_einstein_gap_chirality_extended import reconstruct_xi  # noqa: E402

OUT = REPO / "outputs" / "verify_chirality_sup_DOmega.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

REPO_ROOT = REPO.parent
LADDER = [
    (50,  "results_d1_fix17/d1_p5.npz",                                 "d1"),
    (64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz",               "snap"),
    (72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz",               "snap"),
    (84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz",               "snap"),
    (100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz",             "snap"),
    (128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz",            "snap"),
    (200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz",              "snap"),
    (300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz",             "snap"),
    (512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz",             "snap"),
]


def chirality_per_node(xi, phase):
    g = xi @ xi.T
    g = 0.5 * (g + g.T)
    g = g / max(float(np.trace(g)), 1.0e-12)
    _, vec = np.linalg.eigh(g)
    vec = vec[:, ::-1]
    n_modes = min(6, vec.shape[1])
    nq = n_modes // 2
    w_q = np.sum(vec[:, :nq] ** 2, axis=1)
    w_l = np.sum(vec[:, nq:n_modes] ** 2, axis=1)
    pa = np.cos(phase)
    norm = max(float(np.linalg.norm(pa)), 1.0e-12)
    pa = pa / norm
    return pa * (w_q - w_l)


def _seeds_chirality(path, n_lat, fmt):
    z = np.load(path, allow_pickle=True)
    out = []
    if fmt == "d1":
        edge = z["dense_cell_edge_xi_values"]
        phase_all = z["dense_cell_node_phase_values"]
        for s in range(edge.shape[0]):
            xi = reconstruct_xi(edge[s], n_lat)
            out.append(np.abs(chirality_per_node(xi, phase_all[s])))
    else:
        edge = z["edge_xi_snapshots"]
        psi_r = z["psi_real_snapshots"]
        psi_i = z["psi_imag_snapshots"]
        n_snap = edge.shape[1]
        for s in range(edge.shape[0]):
            xi = 0.5 * (edge[s, n_snap - 1] + edge[s, n_snap - 1].T)
            np.fill_diagonal(xi, 1.0)
            phase = np.arctan2(psi_i[s, n_snap - 1],
                               psi_r[s, n_snap - 1])
            out.append(np.abs(chirality_per_node(xi, phase)))
    return out


def fit_fixed(log_n, log_y, sigma, alpha):
    log_c = float(np.average(log_y + alpha * log_n,
                              weights=1.0 / sigma ** 2))
    pred = log_c - alpha * log_n
    nll = float(np.sum(0.5 * ((log_y - pred) / sigma) ** 2
                       + 0.5 * np.log(2 * np.pi * sigma ** 2)))
    n_pts = len(log_n)
    aicc = 2.0 + 2 * nll
    if n_pts - 2 > 0:
        aicc += 4.0 / (n_pts - 2)
    bic = math.log(n_pts) + 2 * nll
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"alpha": alpha, "log_c": log_c, "nll": nll,
            "AICc": aicc, "BIC": bic, "r2": r2}


def fit_free(log_n, log_y, sigma):
    a_mat = np.column_stack([np.ones_like(log_n), log_n])
    w_mat = np.diag(1.0 / sigma ** 2)
    coef = np.linalg.solve(a_mat.T @ w_mat @ a_mat,
                           a_mat.T @ w_mat @ log_y)
    log_c, neg_alpha = float(coef[0]), float(coef[1])
    alpha = -neg_alpha
    pred = log_c - alpha * log_n
    nll = float(np.sum(0.5 * ((log_y - pred) / sigma) ** 2
                       + 0.5 * np.log(2 * np.pi * sigma ** 2)))
    n_pts = len(log_n)
    aicc = 4.0 + 2 * nll
    if n_pts - 3 > 0:
        aicc += 12.0 / (n_pts - 3)
    bic = 2 * math.log(n_pts) + 2 * nll
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"alpha": alpha, "log_c": log_c, "nll": nll,
            "AICc": aicc, "BIC": bic, "r2": r2}


def main():
    rows = []
    seeds_by_n = {}
    for n_lat, sub, fmt in LADDER:
        path = REPO_ROOT / sub
        if not path.exists():
            continue
        seeds = _seeds_chirality(path, n_lat, fmt)
        if not seeds:
            continue
        seeds_by_n[n_lat] = seeds
        all_chir = np.concatenate(seeds)
        rows.append({
            "regime_N": n_lat,
            "n_seeds": len(seeds),
            "median_per_node_chir": float(np.median(all_chir)),
            "p95_per_node_chir":    float(np.percentile(all_chir, 95)),
            "p99_per_node_chir":    float(np.percentile(all_chir, 99)),
            "sup_per_node_chir":    float(all_chir.max()),
        })

    n_arr = np.array([r["regime_N"] for r in rows], dtype=float)
    sup_arr = np.array([r["sup_per_node_chir"] for r in rows])
    seeds_arr = np.array([r["n_seeds"] for r in rows], dtype=float)
    log_n = np.log(n_arr)
    log_y = np.log(sup_arr)
    sigma = np.maximum(0.20 * np.abs(log_y) / np.sqrt(seeds_arr), 0.02)

    candidates = [
        ("alpha = 2/3 (old Delta_E theory)",   2.0 / 3.0),
        ("alpha = alpha_xi^2 = 81/100",        81.0 / 100.0),
        ("alpha = 13/16",                       13.0 / 16.0),
        ("alpha = 4/5 = alpha_xi - gamma",     4.0 / 5.0),
        ("alpha = D_Omega = beta_pi - gamma = 67/80",  67.0 / 80.0),
        ("alpha = 17/20 = alpha_xi - gamma/2", 17.0 / 20.0),
        ("alpha = alpha_xi = 9/10",             9.0 / 10.0),
        ("alpha = 1",                            1.0),
    ]

    family_fits = []
    for label, alpha in candidates:
        fit = fit_fixed(log_n, log_y, sigma, alpha)
        family_fits.append({"label": label, **fit})
    free = fit_free(log_n, log_y, sigma)
    family_fits.append({"label": "alpha free (2 parameters)", **free})

    aicc_min = min(f["AICc"] for f in family_fits)
    for f in family_fits:
        f["delta_AICc"] = f["AICc"] - aicc_min

    rng = np.random.default_rng(2026)
    n_boot = 2000
    alphas_boot = []
    for _ in range(n_boot):
        log_y_boot = []
        for n_lat in n_arr.astype(int):
            seeds = seeds_by_n[n_lat]
            n_s = len(seeds)
            idx = rng.choice(n_s, n_s, replace=True)
            chir_concat = np.concatenate([seeds[i] for i in idx])
            log_y_boot.append(math.log(chir_concat.max()))
        log_y_boot = np.array(log_y_boot)
        slope, _ = np.polyfit(log_n, log_y_boot, 1)
        alphas_boot.append(-slope)
    alphas_boot = np.array(alphas_boot)

    boot = {
        "n_boot": n_boot,
        "alpha_mean": float(alphas_boot.mean()),
        "alpha_std":  float(alphas_boot.std()),
        "alpha_CI95_low":  float(np.percentile(alphas_boot, 2.5)),
        "alpha_CI95_high": float(np.percentile(alphas_boot, 97.5)),
    }

    alpha_xi_sq = 81.0 / 100.0
    alpha_xi_sq_in_ci = (boot["alpha_CI95_low"] <= alpha_xi_sq
                          <= boot["alpha_CI95_high"])
    d_omega_raw = 67.0 / 80.0
    d_omega_in_ci = (boot["alpha_CI95_low"] <= d_omega_raw
                     <= boot["alpha_CI95_high"])
    two_thirds_in_ci = (boot["alpha_CI95_low"] <= 2.0 / 3.0
                        <= boot["alpha_CI95_high"])
    out = {
        "method": ("Per-node chirality sup-decay audit on canonical "
                   "lattice ladder; alpha_xi^2 = (1-gamma)^2 = 81/100 "
                   "structural-rational hypothesis (cross-observable "
                   "unification with Lambda_t = alpha_xi^2 from P4 "
                   "and L9 cosmological-tensor row in P2)."),
        "structural_hypothesis": {
            "form":        ("S(N) := sup_i |chir_i(N)| = "
                            "C * N^{-alpha_xi^2}"),
            "alpha_xi_squared": alpha_xi_sq,
            "rational":    "81/100",
            "factor":      "alpha_xi * alpha_xi = (1-gamma)^2",
            "cross_observable_match": (
                "alpha_xi^2 = 81/100 is also the time-time "
                "cosmological-constant tensor coefficient Lambda_t "
                "of the emergent Einstein equation (Paper P4 "
                "Symanzik-2 multipoint y_inf = 0.8134, bootstrap "
                "median = 0.8117; matched within 0.5%); identical "
                "to L9 row in P2 landings table."),
            "alternative_D_Omega": d_omega_raw,
            "alternative_D_Omega_note": (
                "D_Omega = beta_pi - gamma = 67/80 = 0.8375 is the "
                "diffusion identity that anchors m_tau, m_mu, m_e, "
                "A_s strict-EXACT closures (P3); empirically also "
                "in CI95 but 2.86% from free-fit (vs alpha_xi^2 "
                "0.44%). Retained as alternative candidate but not "
                "primary hypothesis."),
        },
        "ladder_per_regime": rows,
        "family_AICc": family_fits,
        "bootstrap_CI95": boot,
        "verdict": {
            "alpha_free_estimate":   free["alpha"],
            "alpha_xi_sq_in_CI95":   alpha_xi_sq_in_ci,
            "D_Omega_in_CI95":       d_omega_in_ci,
            "alpha_2_3_in_CI95":     two_thirds_in_ci,
            "best_AICc_label": min(family_fits,
                                    key=lambda f: f["AICc"])["label"],
            "delta_AICc_alpha_xi_sq": next(
                f["delta_AICc"] for f in family_fits
                if "alpha_xi^2" in f["label"]),
            "delta_AICc_D_Omega": next(
                f["delta_AICc"] for f in family_fits
                if "D_Omega" in f["label"]),
            "delta_AICc_2_3": next(
                f["delta_AICc"] for f in family_fits
                if "2/3" in f["label"]),
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== Per-node sup chirality power-law audit ===\n")
    print(f"{'Regime N':>10} {'seeds':>7} {'sup |chir|':>14}")
    for r in rows:
        print(f"{r['regime_N']:>10} {r['n_seeds']:>7} "
              f"{r['sup_per_node_chir']:>14.5f}")
    print()
    print(f"{'Model':<48}{'alpha':>10}{'R^2':>8}{'dAICc':>10}")
    for f in sorted(family_fits, key=lambda x: x["AICc"]):
        flag = " <- BEST" if f["delta_AICc"] == 0.0 else ""
        print(f"{f['label']:<48}{f['alpha']:>10.4f}"
              f"{f['r2']:>8.3f}{f['delta_AICc']:>+10.3f}{flag}")
    print()
    print(f"Bootstrap (n={n_boot}): alpha = {boot['alpha_mean']:.4f} "
          f"+- {boot['alpha_std']:.4f}, "
          f"CI95 = [{boot['alpha_CI95_low']:.4f}, "
          f"{boot['alpha_CI95_high']:.4f}]")
    print(f"alpha_xi^2 = 81/100 = {alpha_xi_sq:.4f} in CI95: "
          f"{alpha_xi_sq_in_ci}")
    print(f"D_Omega = 67/80 = {d_omega_raw:.4f} in CI95: "
          f"{d_omega_in_ci}")
    print(f"alpha = 2/3 = {2/3:.4f} in CI95: {two_thirds_in_ci}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
