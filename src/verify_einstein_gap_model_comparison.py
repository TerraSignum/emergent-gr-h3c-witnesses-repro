"""Reviewer follow-up: model-comparison table for the
chirality-balance five-point fit.

We fit the per-regime deviation $y_n = D(N_n)$ under five
candidate decay laws on the same 5-point data:
  M1: y = C N^{-2/3}                 (theory-target alpha = 2/3)
  M2: y = C N^{-alpha}, alpha free   (free power-law)
  M3: y = C N^{-1}                   (linear decay)
  M4: y = C N^{-2}                   (quadratic decay)
  M5: y = a + b/N + c/N^2            (Symanzik 1/N + 1/N^2)

For each model we report:
  R^2, residual variance, AICc, BIC, leave-one-out RMSE,
and a 1000-sample bootstrap 95% CI on the asymptotic-leading
parameter (alpha for M1-M4, a for M5).

Output: outputs/einstein_gap_model_comparison.json
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "data" / "einstein_gap_5point_fit.json"
OUT = REPO / "outputs" / "einstein_gap_model_comparison.json"


def _load():
    with open(IN, encoding="utf-8") as f:
        return json.load(f)


def _fit_loglog_alpha(xs, ys):
    log_x = [math.log(x) for x in xs]
    log_y = [math.log(y) for y in ys]
    n = len(xs)
    sx = sum(log_x); sy = sum(log_y)
    sxx = sum(lx * lx for lx in log_x)
    sxy = sum(lx * ly for lx, ly in zip(log_x, log_y))
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    pred = [slope * lx + intercept for lx in log_x]
    ss_res = sum((ly - p) ** 2 for ly, p in zip(log_y, pred))
    ss_tot = sum((ly - sy / n) ** 2 for ly in log_y)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return -slope, math.exp(intercept), r2, ss_res


def _fit_fixed_alpha(xs, ys, alpha):
    log_x = [math.log(x) for x in xs]
    log_y = [math.log(y) for y in ys]
    log_C = [ly + alpha * lx for ly, lx in zip(log_y, log_x)]
    log_C_mean = sum(log_C) / len(log_C)
    pred = [-alpha * lx + log_C_mean for lx in log_x]
    ss_res = sum((ly - p) ** 2 for ly, p in zip(log_y, pred))
    ss_tot = sum((ly - sum(log_y) / len(log_y)) ** 2 for ly in log_y)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return alpha, math.exp(log_C_mean), r2, ss_res


def _fit_symanzik24(xs, ys):
    n = len(xs)
    rows = []
    for x, y in zip(xs, ys):
        rows.append([1.0, 1.0 / x, 1.0 / (x * x), y])
    aa = [[0.0, 0.0, 0.0] for _ in range(3)]
    bb = [0.0, 0.0, 0.0]
    for r in rows:
        for i in range(3):
            for j in range(3):
                aa[i][j] += r[i] * r[j]
            bb[i] += r[i] * r[3]

    def solve(a, b):
        # 3x3 Gaussian elimination
        a = [row[:] for row in a]
        b = b[:]
        for i in range(3):
            piv = a[i][i]
            for j in range(i, 3):
                a[i][j] /= piv
            b[i] /= piv
            for k in range(3):
                if k != i:
                    factor = a[k][i]
                    for j in range(i, 3):
                        a[k][j] -= factor * a[i][j]
                    b[k] -= factor * b[i]
        return b
    coef = solve(aa, bb)
    a_inf, b_coef, c_coef = coef
    pred = [a_inf + b_coef / x + c_coef / (x * x) for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - sum(ys) / n) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a_inf, b_coef, c_coef, r2, ss_res


def _aicc_bic(ss_res, n, k):
    if ss_res <= 0 or n <= k + 1:
        return float("inf"), float("inf")
    sigma2 = ss_res / n
    aic = n * math.log(sigma2) + 2.0 * k
    aicc = aic + 2.0 * k * (k + 1) / (n - k - 1)
    bic = n * math.log(sigma2) + k * math.log(n)
    return aicc, bic


def _loo_rmse(xs, ys, fit_fn):
    n = len(xs)
    sq_err = 0.0
    for i in range(n):
        x_train = xs[:i] + xs[i+1:]
        y_train = ys[:i] + ys[i+1:]
        pred = fit_fn(x_train, y_train, xs[i])
        sq_err += (ys[i] - pred) ** 2
    return math.sqrt(sq_err / n)


def _predict_alpha_free(x_train, y_train, x_test):
    a, c, _, _ = _fit_loglog_alpha(x_train, y_train)
    return c * x_test ** (-a)


def _predict_alpha_fixed(alpha):
    def f(x_train, y_train, x_test):
        _, c, _, _ = _fit_fixed_alpha(x_train, y_train, alpha)
        return c * x_test ** (-alpha)
    return f


def _predict_symanzik24(x_train, y_train, x_test):
    a, b, c, _, _ = _fit_symanzik24(x_train, y_train)
    return a + b / x_test + c / (x_test * x_test)


def _bootstrap_alpha(xs, ys, n_boot=1000, alpha=None):
    rng = random.Random(0xC0FFEE)
    n = len(xs)
    samples = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        x_b = [xs[i] for i in idx]
        y_b = [ys[i] for i in idx]
        try:
            if alpha is None:
                a, *_ = _fit_loglog_alpha(x_b, y_b)
            else:
                a = alpha
            samples.append(a)
        except Exception:
            continue
    samples.sort()
    if not samples:
        return None
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples))]
    return [lo, hi]


def _bootstrap_a_inf(xs, ys, n_boot=1000):
    rng = random.Random(0xC0FFEE)
    n = len(xs)
    samples = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        x_b = [xs[i] for i in idx]
        y_b = [ys[i] for i in idx]
        try:
            a, *_ = _fit_symanzik24(x_b, y_b)
            samples.append(a)
        except Exception:
            continue
    samples.sort()
    if not samples:
        return None
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples))]
    return [lo, hi]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = _load()
    xs = list(d["scan"]["N_values"])
    ys = list(d["scan"]["deviation_means"])
    n = len(xs)

    rows = []
    # M1: alpha = 2/3 fixed
    a1, c1, r2_1, ss1 = _fit_fixed_alpha(xs, ys, 2.0 / 3.0)
    aicc1, bic1 = _aicc_bic(ss1, n, 1)
    loo1 = _loo_rmse(xs, ys, _predict_alpha_fixed(2.0 / 3.0))
    rows.append({"model": "M1: alpha = 2/3 (theory)",
                  "alpha": a1, "C": c1, "R2": r2_1,
                  "ss_res": ss1, "AICc": aicc1, "BIC": bic1,
                  "LOO_RMSE": loo1, "k_params": 1,
                  "alpha_bootstrap_95CI": _bootstrap_alpha(xs, ys, alpha=2.0/3.0)})
    # M2: alpha free
    a2, c2, r2_2, ss2 = _fit_loglog_alpha(xs, ys)
    aicc2, bic2 = _aicc_bic(ss2, n, 2)
    loo2 = _loo_rmse(xs, ys, _predict_alpha_free)
    rows.append({"model": "M2: alpha free", "alpha": a2, "C": c2,
                  "R2": r2_2, "ss_res": ss2, "AICc": aicc2,
                  "BIC": bic2, "LOO_RMSE": loo2, "k_params": 2,
                  "alpha_bootstrap_95CI": _bootstrap_alpha(xs, ys)})
    # M3: alpha = 1
    a3, c3, r2_3, ss3 = _fit_fixed_alpha(xs, ys, 1.0)
    aicc3, bic3 = _aicc_bic(ss3, n, 1)
    loo3 = _loo_rmse(xs, ys, _predict_alpha_fixed(1.0))
    rows.append({"model": "M3: alpha = 1", "alpha": a3, "C": c3,
                  "R2": r2_3, "ss_res": ss3, "AICc": aicc3,
                  "BIC": bic3, "LOO_RMSE": loo3, "k_params": 1})
    # M4: alpha = 2
    a4, c4, r2_4, ss4 = _fit_fixed_alpha(xs, ys, 2.0)
    aicc4, bic4 = _aicc_bic(ss4, n, 1)
    loo4 = _loo_rmse(xs, ys, _predict_alpha_fixed(2.0))
    rows.append({"model": "M4: alpha = 2", "alpha": a4, "C": c4,
                  "R2": r2_4, "ss_res": ss4, "AICc": aicc4,
                  "BIC": bic4, "LOO_RMSE": loo4, "k_params": 1})
    # M5: Symanzik 1/N + 1/N^2
    a_inf, b_c, c_c, r2_5, ss5 = _fit_symanzik24(xs, ys)
    aicc5, bic5 = _aicc_bic(ss5, n, 3)
    loo5 = _loo_rmse(xs, ys, _predict_symanzik24)
    rows.append({"model": "M5: a + b/N + c/N^2 (Symanzik 2+4)",
                  "a_inf": a_inf, "b": b_c, "c": c_c,
                  "R2": r2_5, "ss_res": ss5, "AICc": aicc5,
                  "BIC": bic5, "LOO_RMSE": loo5, "k_params": 3,
                  "a_inf_bootstrap_95CI": _bootstrap_a_inf(xs, ys)})

    out = {
        "method": "model comparison on the 5-point chirality-balance fit",
        "input_file": str(IN.relative_to(REPO)),
        "N_values": xs,
        "deviation_means": ys,
        "models": rows,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print()
    print(f"{'model':<40s} {'R^2':>8s} {'AICc':>9s} {'BIC':>9s} "
          f"{'LOO_RMSE':>11s}")
    print("-" * 90)
    for r in rows:
        print(f"{r['model']:<40s} {r['R2']:>8.4f} {r['AICc']:>9.3f} "
              f"{r['BIC']:>9.3f} {r['LOO_RMSE']:>11.5f}")
    print()
    delta_aicc = sorted((m["AICc"], m["model"]) for m in rows)
    print(f"Best by AICc: {delta_aicc[0][1]} (AICc {delta_aicc[0][0]:.3f}); "
          f"delta_AICc to M1: {rows[0]['AICc'] - delta_aicc[0][0]:+.3f}")


if __name__ == "__main__":
    main()
