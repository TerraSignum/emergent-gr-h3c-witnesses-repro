"""Generate the three impactful figures for the H3C witnesses paper:
  fig_chirality_5pt_loglog.pdf
  fig_Rbar_9pt_loglog.pdf
  fig_chi_sup_decay.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless CI / no-DISPLAY
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (vector, arXiv-friendly)
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def fig_chirality_5pt():
    p = REPO / "data" / "einstein_gap_5point_fit.json"
    with open(p) as f:
        d = json.load(f)
    scan = d.get("scan", {})
    fit = d.get("fit_result", {})
    Ns = np.array(scan.get("N_values", []), dtype=float)
    ys = np.array(scan.get("deviation_means", []), dtype=float)
    if len(Ns) == 0:
        return
    alpha = float(fit.get("alpha_fit", 0.6355))
    r2 = float(fit.get("r2_loglog", 0.78))

    mask = ys > 0
    Ns_p = Ns[mask]; ys_p = ys[mask]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.loglog(Ns_p, ys_p, "o", markersize=8, color="#3c6ea7",
              markeredgecolor="black", label="5-pt within-P5 chirality data")
    if len(Ns_p) >= 2:
        c0 = np.exp(np.mean(np.log(ys_p) + alpha * np.log(Ns_p)))
        xs = np.logspace(np.log10(Ns_p.min()), np.log10(Ns_p.max()), 50)
        ax.loglog(xs, c0 * xs ** (-alpha), color="#d97f4a", linewidth=1.4,
                  label=f"$\\hat\\alpha={alpha:.4f}$, $R^2={r2:.2f}$")
    ax.set_xlabel(r"$N$"); ax.set_ylabel(r"$\delta_c(N)$")
    ax.set_title("5-pt within-P5 chirality residual log-log fit")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = OUT / "fig_chirality_5pt_loglog.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO)}")


def fig_Rbar_9pt():
    p = REPO / "data" / "einstein_gap_9point_witnesses.json"
    with open(p) as f:
        d = json.load(f)
    Ns = np.array(d["lattice_ladder"]["N_values"], dtype=float)
    Rbar = np.array(d["primary_curvature_side_witness"]["values"], dtype=float)
    mask = Rbar > 0
    Ns_p = Ns[mask]; R_p = Rbar[mask]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.loglog(Ns_p, R_p, "o", markersize=8, color="#3c6ea7",
              markeredgecolor="black", label="9-pt $\\bar R(N)$")
    # alpha=2/3 line
    c_fix = np.exp(np.mean(np.log(R_p) + (2/3)*np.log(Ns_p)))
    xs = np.logspace(np.log10(Ns_p.min()), np.log10(Ns_p.max()), 50)
    ax.loglog(xs, c_fix * xs**(-2/3), color="#d97f4a", linewidth=1.4,
              linestyle="--", label="$\\alpha=2/3$ ($R^2=0.83$)")
    # free alpha = 0.843
    c_free = np.exp(np.mean(np.log(R_p) + 0.843 * np.log(Ns_p)))
    ax.loglog(xs, c_free * xs**(-0.843), color="#5a3010", linewidth=1.4,
              label="free $\\alpha=0.843$ ($R^2=0.87$)")
    ax.set_xlabel(r"$N$"); ax.set_ylabel(r"$\bar R(N)$")
    ax.set_title("9-pt mean Ricci-scalar witness log-log")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = OUT / "fig_Rbar_9pt_loglog.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO)}")


def fig_chi_sup_decay():
    p = REPO / "outputs" / "verify_chirality_sup_DOmega.json"
    with open(p) as f:
        d = json.load(f)
    ladder = d.get("ladder_per_regime", [])
    Ns = np.array([r.get("regime_N", 0) for r in ladder], dtype=float)
    ys = np.array([r.get("sup_per_node_chir", 0) for r in ladder], dtype=float)
    if len(Ns) == 0:
        return
    fam = d.get("family_AICc", {})
    M2 = fam.get("M2_free_alpha", {}) if isinstance(fam, dict) else {}
    alpha_free = float(M2.get("alpha", 0.8136))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    mask = ys > 0
    Ns_p = Ns[mask]; ys_p = ys[mask]
    ax.loglog(Ns_p, ys_p, "o", markersize=8, color="#3c6ea7",
              markeredgecolor="black", label="$\\chi_i$ sup-decay")
    c0 = np.exp(np.mean(np.log(ys_p) + alpha_free * np.log(Ns_p)))
    xs = np.logspace(np.log10(Ns_p.min()), np.log10(Ns_p.max()), 50)
    ax.loglog(xs, c0 * xs**(-alpha_free), color="#d97f4a", linewidth=1.4,
              label=f"free $\\hat\\alpha={alpha_free:.4f}$")
    # alpha_xi^2 = 0.81 reference
    c_axi2 = np.exp(np.mean(np.log(ys_p) + 0.81*np.log(Ns_p)))
    ax.loglog(xs, c_axi2 * xs**(-0.81), color="#5a3010", linewidth=1.0,
              linestyle=":", label=r"$\alpha_\xi^2=0.81$ reference")
    ax.set_xlabel(r"$N$"); ax.set_ylabel(r"$\chi_i^{\sup}(N)$")
    ax.set_title("Per-direction chirality sup-decay log-log")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = OUT / "fig_chi_sup_decay.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO)}")


def main():
    for f in (fig_chirality_5pt, fig_Rbar_9pt, fig_chi_sup_decay):
        try:
            f()
        except Exception as e:
            print(f"{f.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
