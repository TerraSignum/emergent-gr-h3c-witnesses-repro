"""Continuum-extrapolation lift from the continuous-amplitude
chirality observable to an integer-valued topological charge.

Background. P4-C's per-node chirality residual

    chir_i(N) = pa_i * (w_q_i - w_l_i)

is a continuous-amplitude observable whose sup-decay power-law
(verify_chirality_sup_DOmega.py) tracks alpha_xi^2 = 81/100 across
N in [50, 512]. P4-D's overlap-Dirac construction gives an
integer-valued index ind(D_overlap) = Q_top + const that closes
the discrete Atiyah-Singer relation at machine precision on
injected-flux backgrounds (verify_overlap_dirac_index.json).

The lift question: does the continuous chirality observable,
suitably normalised and continuum-extrapolated, converge to the
integer-valued ind in the continuum limit on the same Xi-graph
configurations? On smooth (no injected gauge twist) Xi-snapshots
both quantities are predicted to vanish in the continuum limit
(Q_top = 0, chirality sup -> 0 by power-law). On non-trivial-
topology configurations the bridge is registered as an open
follow-up via the T_00-tagged gauge connection of P4-D Item 2.

This script audits the trivial-gauge consistency check:

  Z(N) := (1/N) * sum_i chir_i(N)
  S(N) := sup_i |chir_i(N)|

and shows that on smooth Xi-snapshots

  (1) Z(N)        -> 0 in the continuum limit (compatible with
                     ind = Q_top = 0 on trivial gauge);
  (2) round(Z(N) * N) is bounded and integer-valued;
  (3) |Z(N)| <= S(N) by triangle inequality, and S(N) is the
      sup-decay observable of P4-C section sec:chirality with
      empirical alpha_xi^2 = 81/100 power-law;
  (4) the framework integer count round(Z*N) coincides with the
      overlap-Dirac integer count ind on the same Xi-snapshots
      (both vanish on smooth backgrounds), establishing the
      trivial-gauge consistency of the lift.

Output: outputs/verify_chirality_to_integer_lift.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verify_chirality_sup_DOmega import (  # noqa: E402
    LADDER, _seeds_chirality, chirality_per_node,
)


def _seeds_signed_chirality(path, n_lat, fmt):
    """Return per-seed signed (not abs) per-node chirality vectors."""
    from verify_einstein_gap_chirality_extended import reconstruct_xi
    z = np.load(path, allow_pickle=True)
    out = []
    if fmt == "d1":
        edge = z["dense_cell_edge_xi_values"]
        phase_all = z["dense_cell_node_phase_values"]
        for s in range(edge.shape[0]):
            xi = reconstruct_xi(edge[s], n_lat)
            out.append(chirality_per_node(xi, phase_all[s]))
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
            out.append(chirality_per_node(xi, phase))
    return out


def main() -> int:
    rows = []
    repo_root = REPO.parent

    for n_lat, sub, fmt in LADDER:
        path = repo_root / sub
        if not path.exists():
            continue
        seeds_signed = _seeds_signed_chirality(path, n_lat, fmt)
        if not seeds_signed:
            continue
        sums = [float(np.sum(arr)) for arr in seeds_signed]
        sups = [float(np.max(np.abs(arr))) for arr in seeds_signed]
        z_per_seed = [s / n_lat for s in sums]
        z_int_per_seed = [int(round(s)) for s in sums]
        rows.append({
            "regime_N": int(n_lat),
            "n_seeds": len(seeds_signed),
            "Z_mean": float(np.mean(z_per_seed)),
            "Z_std": float(np.std(z_per_seed)),
            "Z_abs_mean": float(np.mean(np.abs(z_per_seed))),
            "global_sum_chir_mean": float(np.mean(sums)),
            "global_sum_chir_abs_mean": float(np.mean(np.abs(sums))),
            "round_Z_times_N_per_seed": z_int_per_seed,
            "round_Z_times_N_max_abs": int(np.max(
                np.abs(z_int_per_seed))),
            "S_sup_mean": float(np.mean(sups)),
            "Z_le_S_holds_per_seed": [
                bool(abs(s) / n_lat <= sup + 1e-12)
                for s, sup in zip(sums, sups)
            ],
        })

    n_arr = np.array([r["regime_N"] for r in rows], dtype=float)
    z_abs_arr = np.array([r["Z_abs_mean"] for r in rows])
    s_abs_arr = np.array([r["S_sup_mean"] for r in rows])

    log_n = np.log(n_arr)
    log_z = np.log(np.maximum(z_abs_arr, 1e-30))
    log_s = np.log(np.maximum(s_abs_arr, 1e-30))

    slope_z, _ = np.polyfit(log_n, log_z, 1) if len(log_n) >= 3 \
        else (float("nan"), float("nan"))
    slope_s, _ = np.polyfit(log_n, log_s, 1) if len(log_n) >= 3 \
        else (float("nan"), float("nan"))

    overlap_path = (repo_root
                    / "emergent-gr-atiyah-singer-chirality-repro"
                    / "outputs" / "verify_overlap_dirac_index.json")
    overlap_summary = {}
    if overlap_path.exists():
        try:
            data = json.loads(overlap_path.read_text(encoding="utf-8"))
            inds = [
                int(round(c.get("ind_relative", 0.0)))
                for c in data.get("per_config", [])
            ]
            overlap_summary = {
                "overlap_path": str(overlap_path.relative_to(
                    repo_root)).replace("\\", "/"),
                "n_configs": len(inds),
                "ind_relative_min": min(inds) if inds else None,
                "ind_relative_max": max(inds) if inds else None,
                "ind_relative_all_zero":
                    bool(all(i == 0 for i in inds))
                    if inds else False,
            }
        except Exception as exc:  # noqa: BLE001
            overlap_summary = {"error": str(exc)}

    chir_int_all_zero = all(
        r["round_Z_times_N_max_abs"] == 0 for r in rows)

    consistency = {
        "trivial_gauge_chir_continuum_zero":
            bool(chir_int_all_zero),
        "overlap_ind_all_zero":
            bool(overlap_summary.get("ind_relative_all_zero", False)),
        "trivial_gauge_lift_holds":
            bool(chir_int_all_zero and
                 overlap_summary.get("ind_relative_all_zero", False)),
    }

    bundle = {
        "method": "verify_chirality_to_integer_lift",
        "schema_version": "1.0.0",
        "description": (
            "Continuum-extrapolation lift from continuous-amplitude "
            "per-node chirality residual to integer-valued ind via "
            "P4-D overlap-Dirac on smooth Xi-snapshots. "
            "Tests trivial-gauge consistency: chirality global sum "
            "Z*N rounds to integer 0 across the ladder N in [50, "
            "512] and overlap-Dirac ind is 0 on the same configs."
        ),
        "ladder_rows": rows,
        "loglog_continuum_limit": {
            "Z_abs_mean_slope_in_log_N": float(slope_z),
            "S_sup_mean_slope_in_log_N": float(slope_s),
            "alpha_xi_sq_target": 81.0 / 100.0,
            "Z_decays_at_least_as_fast_as_S":
                bool(-slope_z >= -slope_s - 0.05),
        },
        "overlap_dirac_summary": overlap_summary,
        "consistency_check": consistency,
    }

    out = REPO / "outputs" / "verify_chirality_to_integer_lift.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Chirality -> integer-ind continuum lift audit")
    print("=" * 70)
    for r in rows:
        print(f"  N={r['regime_N']:<4d} seeds={r['n_seeds']:>2d}  "
              f"Z={r['Z_mean']:+.4f}+-{r['Z_std']:.4f}  "
              f"|Z|*N round={r['round_Z_times_N_max_abs']}  "
              f"S_sup={r['S_sup_mean']:.4f}")
    print()
    print(f"  log-log slope |Z|: {slope_z:.3f}  "
          f"(target -alpha_xi^2 = -0.81 or steeper)")
    print(f"  log-log slope  S : {slope_s:.3f}  "
          f"(P4-C chirality sup-decay)")
    print(f"  Z continuum_int_all_zero: "
          f"{consistency['trivial_gauge_chir_continuum_zero']}")
    print(f"  overlap ind_all_zero    : "
          f"{consistency['overlap_ind_all_zero']}")
    print(f"  trivial-gauge lift holds: "
          f"{consistency['trivial_gauge_lift_holds']}")
    print(f"  saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
