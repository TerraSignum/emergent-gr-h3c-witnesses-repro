"""Alpha-exponent model comparison for the chirality witness.

Tests five alpha-models on the chirality-balance log-log fit
on the within-canonical-regime ladder
N in {50, 64, 72, 84, 100, 128, 200, 300}:
  M1: alpha = 2/3                  (theory-fixed)
  M2: alpha free                   (single-parameter fit)
  M3: alpha = 1                    (linear)
  M4: alpha = 2                    (quadratic)
  M5: Symanzik-2  D = a + b/N + c/N^2  (two-parameter, alpha-free)

For each model: log-likelihood (Gaussian residual under the
binned standard error), AICc, BIC, and leave-one-out (LOO)
predictive log-likelihood.

Output: outputs/verify_alpha_model_comparison.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "verify_alpha_model_comparison.json"

# Within-canonical-regime chirality data: load from bundled JSON
# (data/einstein_gap_chirality_within_p5.json).
_DATA_PATH = REPO / "data" / "einstein_gap_chirality_within_p5.json"
_d = json.loads(_DATA_PATH.read_text())
LADDER_N = np.array(_d["N_values"], dtype=float)
D_VALUES = np.array(_d["deviation_means"], dtype=float)
# Per-seed-count weighted standard error: sigma_i ~ D_i / sqrt(n_seeds_i)
# (Gaussian approximation under the per-seed-mean estimator).
_n_seeds = np.array([row["n_seeds"] for row in _d["rows"]], dtype=float)
SIGMA = np.maximum(0.20 * D_VALUES / np.sqrt(_n_seeds), 1e-4)


def neg_log_likelihood_gaussian(y, y_pred, sigma):
    """Sum_i [-0.5 log(2 pi sigma_i^2) - (y_i - y_pred_i)^2 / (2 sigma_i^2)]."""
    return float(np.sum(0.5 * ((y - y_pred) / sigma) ** 2
                         + 0.5 * np.log(2 * np.pi * sigma**2)))


def aicc(nll, n, k):
    """Standard AICc with negative-log-likelihood input.
    AIC = 2k - 2 log L = 2k + 2 nll. Corrected for small samples."""
    aic = 2 * k + 2 * nll
    if n - k - 1 > 0:
        return aic + 2 * k * (k + 1) / (n - k - 1)
    return float("nan")


def bic_(nll, n, k):
    """BIC = k ln n - 2 log L = k ln n + 2 nll."""
    return k * math.log(n) + 2 * nll


def fit_log_log_with_fixed_alpha(N, D, alpha, sigma):
    """D = c * N^{-alpha} -> log D = log c - alpha log N. One free
    parameter c."""
    log_N = np.log(N)
    log_D = np.log(D)
    # Weighted regression for log c with fixed alpha
    # log D + alpha log N = log c
    log_c = float(np.average(log_D + alpha * log_N,
                              weights=1.0 / sigma**2))
    log_pred = log_c - alpha * log_N
    pred = np.exp(log_pred)
    # Residual sum of squares in original D-space
    nll = neg_log_likelihood_gaussian(D, pred, sigma)
    k = 1
    return {
        "alpha": alpha, "log_c": log_c,
        "n_params": k,
        "neg_log_likelihood": nll,
        "AICc": aicc(nll, len(D), k),
        "BIC": bic_(nll, len(D), k),
        "predictions": pred.tolist(),
    }


def fit_log_log_free_alpha(N, D, sigma):
    """Free-alpha log-log fit: log D = log c - alpha log N.
    Two free parameters: log c, alpha."""
    log_N = np.log(N)
    log_D = np.log(D)
    A = np.column_stack([np.ones(len(N)), log_N])
    W = np.diag(1.0 / sigma**2)
    coef = np.linalg.solve(A.T @ W @ A, A.T @ W @ log_D)
    log_c, neg_alpha = float(coef[0]), float(coef[1])
    alpha = -neg_alpha
    log_pred = log_c - alpha * log_N
    pred = np.exp(log_pred)
    nll = neg_log_likelihood_gaussian(D, pred, sigma)
    k = 2
    return {
        "alpha": alpha, "log_c": log_c,
        "n_params": k,
        "neg_log_likelihood": nll,
        "AICc": aicc(nll, len(D), k),
        "BIC": bic_(nll, len(D), k),
        "predictions": pred.tolist(),
    }


def fit_symanzik_2(N, D, sigma):
    """D = a + b/N + c/N^2. Three parameters."""
    inv_N = 1.0 / N
    inv_N2 = 1.0 / N**2
    A = np.column_stack([np.ones(len(N)), inv_N, inv_N2])
    W = np.diag(1.0 / sigma**2)
    coef = np.linalg.solve(A.T @ W @ A, A.T @ W @ D)
    a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
    pred = a + b * inv_N + c * inv_N2
    nll = neg_log_likelihood_gaussian(D, pred, sigma)
    k = 3
    return {
        "a": a, "b": b, "c": c,
        "n_params": k,
        "neg_log_likelihood": nll,
        "AICc": aicc(nll, len(D), k),
        "BIC": bic_(nll, len(D), k),
        "asymptote": a,
        "predictions": pred.tolist(),
    }


def loo_log_likelihood(N, D, sigma, fit_fn):
    """Leave-one-out predictive log-likelihood: refit on N\{i}, predict
    point i, sum log p(D_i | model_-i)."""
    n = len(N)
    log_lik_total = 0.0
    for i in range(n):
        keep = np.array([j for j in range(n) if j != i])
        N_train = N[keep]; D_train = D[keep]; s_train = sigma[keep]
        try:
            fit = fit_fn(N_train, D_train, s_train)
        except Exception:  # noqa: BLE001
            return float("nan")
        # Predict at held-out point
        if "alpha" in fit and "log_c" in fit:
            pred_i = float(np.exp(fit["log_c"] - fit["alpha"] * np.log(N[i])))
        elif "a" in fit and "b" in fit and "c" in fit:
            pred_i = fit["a"] + fit["b"] / N[i] + fit["c"] / N[i]**2
        else:
            return float("nan")
        log_lik_total += -0.5 * ((D[i] - pred_i) / sigma[i])**2 \
                          - 0.5 * math.log(2 * math.pi * sigma[i]**2)
    return float(log_lik_total)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    N, D, sigma = LADDER_N, D_VALUES, SIGMA

    fits = {}
    fits["M1_alpha_2_over_3"] = fit_log_log_with_fixed_alpha(N, D, 2/3, sigma)
    fits["M2_alpha_free"]     = fit_log_log_free_alpha(N, D, sigma)
    fits["M3_alpha_1"]        = fit_log_log_with_fixed_alpha(N, D, 1.0, sigma)
    fits["M4_alpha_2"]        = fit_log_log_with_fixed_alpha(N, D, 2.0, sigma)
    fits["M5_symanzik_2"]     = fit_symanzik_2(N, D, sigma)

    # LOO log-likelihood for each model
    fits["M1_alpha_2_over_3"]["LOO_log_lik"] = loo_log_likelihood(
        N, D, sigma, lambda N_, D_, s_: fit_log_log_with_fixed_alpha(N_, D_, 2/3, s_))
    fits["M2_alpha_free"]["LOO_log_lik"] = loo_log_likelihood(
        N, D, sigma, lambda N_, D_, s_: fit_log_log_free_alpha(N_, D_, s_))
    fits["M3_alpha_1"]["LOO_log_lik"] = loo_log_likelihood(
        N, D, sigma, lambda N_, D_, s_: fit_log_log_with_fixed_alpha(N_, D_, 1.0, s_))
    fits["M4_alpha_2"]["LOO_log_lik"] = loo_log_likelihood(
        N, D, sigma, lambda N_, D_, s_: fit_log_log_with_fixed_alpha(N_, D_, 2.0, s_))
    fits["M5_symanzik_2"]["LOO_log_lik"] = loo_log_likelihood(
        N, D, sigma, fit_symanzik_2)

    # Find AICc-best
    aiccs = {k: v["AICc"] for k, v in fits.items() if math.isfinite(v["AICc"])}
    best = min(aiccs, key=aiccs.get)
    a_min = aiccs[best]
    for k, v in fits.items():
        if math.isfinite(v["AICc"]):
            v["delta_AICc"] = v["AICc"] - a_min

    out = {
        "method": "Alpha-exponent model comparison for chirality witness",
        "ladder_N": LADDER_N.tolist(),
        "D_values": D_VALUES.tolist(),
        "sigma": SIGMA.tolist(),
        "models": {
            "M1": "alpha = 2/3 (theory-fixed)",
            "M2": "alpha free (single-parameter)",
            "M3": "alpha = 1 (linear)",
            "M4": "alpha = 2 (quadratic)",
            "M5": "Symanzik-2 (D = a + b/N + c/N^2, alpha-free)",
        },
        "fits": fits,
        "AICc_best_model": best,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== Alpha model comparison ===")
    print(f"{'Model':<25s} {'k':>3s}  {'AICc':>10s} {'BIC':>10s} "
          f"{'dAICc':>8s} {'LOO ll':>10s}")
    for k, v in sorted(fits.items(), key=lambda x: x[1]["AICc"]):
        cell_d = f"{v.get('delta_AICc', float('nan')):+8.2f}"
        cell_loo = f"{v['LOO_log_lik']:10.2f}" if math.isfinite(
            v["LOO_log_lik"]) else "      ----"
        print(f"{k:<25s} {v['n_params']:>3d}  "
              f"{v['AICc']:>10.2f} {v['BIC']:>10.2f} {cell_d:>8s} {cell_loo}")
    print(f"\nAICc-best: {best}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
