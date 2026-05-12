"""Re-fit R-bar(N) on the 9-point ladder with free alpha + AICc comparison
to alpha=2/3, output certificate at outputs/einstein_gap_Rbar_free_alpha.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main():
    p = REPO / "data" / "einstein_gap_9point_witnesses.json"
    with open(p, encoding="utf-8") as f:
        d = json.load(f)

    Ns = np.array(d["lattice_ladder"]["N_values"], dtype=float)
    Rbar = np.array(d["primary_curvature_side_witness"]["values"],
                     dtype=float)

    # Filter positive only (log-log)
    mask = Rbar > 0
    Ns_f = Ns[mask]
    Rbar_f = Rbar[mask]

    # Free alpha fit: log(Rbar) = log(C) - alpha * log(N)
    logN = np.log(Ns_f)
    logR = np.log(Rbar_f)
    A = np.column_stack([np.ones_like(logN), -logN])
    coef, *_ = np.linalg.lstsq(A, logR, rcond=None)
    logC, alpha_free = float(coef[0]), float(coef[1])
    pred = A @ coef
    rss = float(np.sum((logR - pred) ** 2))
    tss = float(np.sum((logR - logR.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    n = len(logR)
    aicc_free = n * np.log(rss / n) + 2 * 2 + 2 * 2 * 3 / max(1, n - 3)

    # Fixed alpha = 2/3
    A_fix = -2.0 / 3.0 * logN
    A_only_C = np.ones_like(logN)
    M = np.column_stack([A_only_C])
    coef_fix, *_ = np.linalg.lstsq(M, logR - A_fix, rcond=None)
    logC_fix = float(coef_fix[0])
    pred_fix = logC_fix + A_fix
    rss_fix = float(np.sum((logR - pred_fix) ** 2))
    r2_fix = 1.0 - rss_fix / tss if tss > 0 else float("nan")
    aicc_fix = n * np.log(rss_fix / n) + 2 * 1 + 2 * 1 * 2 / max(1, n - 2)

    # Skip-N=18: drop smallest N
    idx_min = np.argmin(Ns_f)
    mask2 = np.ones(n, dtype=bool); mask2[idx_min] = False
    logN2 = logN[mask2]; logR2 = logR[mask2]
    A2 = np.column_stack([np.ones_like(logN2), -logN2])
    coef2, *_ = np.linalg.lstsq(A2, logR2, rcond=None)
    alpha_free_skip = float(coef2[1])
    pred2 = A2 @ coef2
    rss2 = float(np.sum((logR2 - pred2) ** 2))
    tss2 = float(np.sum((logR2 - logR2.mean()) ** 2))
    r2_skip = 1.0 - rss2 / tss2 if tss2 > 0 else float("nan")

    out = {
        "method": "Rbar_free_alpha_AICc",
        "stand": "2026-05-04",
        "data_source": "data/einstein_gap_9point_witnesses.json",
        "n_points_full": n,
        "free_alpha_full": {
            "alpha": alpha_free,
            "log_C": logC,
            "R2": r2,
            "rss": rss,
            "aicc": aicc_free,
        },
        "fixed_alpha_2_3_full": {
            "alpha": 2.0 / 3.0,
            "log_C": logC_fix,
            "R2": r2_fix,
            "rss": rss_fix,
            "aicc": aicc_fix,
        },
        "free_alpha_skip_N18": {
            "alpha": alpha_free_skip,
            "R2": r2_skip,
            "n_points": int(mask2.sum()),
        },
        "delta_aicc_free_minus_fixed": aicc_free - aicc_fix,
    }
    out_path = REPO / "outputs" / "einstein_gap_Rbar_free_alpha.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")
    print(f"  free alpha (full 9pt) = {alpha_free:.4f}, R^2={r2:.3f}, AICc={aicc_free:.2f}")
    print(f"  fixed 2/3   (full 9pt) = R^2={r2_fix:.3f}, AICc={aicc_fix:.2f}")
    print(f"  free alpha (skip N=18) = {alpha_free_skip:.4f}, R^2={r2_skip:.3f}")


if __name__ == "__main__":
    main()
