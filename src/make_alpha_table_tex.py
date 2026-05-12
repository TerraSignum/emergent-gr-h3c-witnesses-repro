"""LaTeX table generator for chirality-sup alpha-family comparison.

Reads the per-node sup-decay audit
(outputs/verify_chirality_sup_DOmega.json) which compares the
candidate exponent family
{2/3, 4/5, 13/16, D_Omega = 67/80, 17/20, alpha_xi = 9/10, 1, free}
on the eight-point seed-corrected ladder. AICc-best is highlighted
in bold; D_Omega-row is also emphasised because it is the
structural-rational hypothesis adopted in the manuscript.
"""
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JSON = REPO / "outputs" / "verify_chirality_sup_DOmega.json"
OUT = REPO / "paper" / "tables" / "tab_alpha_model_comparison.tex"
OUT.parent.mkdir(parents=True, exist_ok=True)

LABEL_TEX = {
    "alpha = 2/3 (old Delta_E theory)":
        r"$\alpha\!=\!2/3$ (Theorem~15.18 P2 bound)",
    "alpha = alpha_xi^2 = 81/100":
        r"$\alpha\!=\!\alpha_{\xi}^{2}\!=\!81/100\!=\!\Lambda_{t}$",
    "alpha = 13/16":
        r"$\alpha\!=\!13/16$",
    "alpha = 4/5 = alpha_xi - gamma":
        r"$\alpha\!=\!4/5\!=\!\alpha_{\xi}\!-\!\gamma$",
    "alpha = D_Omega = beta_pi - gamma = 67/80":
        r"$\alpha\!=\!D_{\Omega}\!=\!\beta_{\pi}\!-\!\gamma\!=\!67/80$",
    "alpha = 17/20 = alpha_xi - gamma/2":
        r"$\alpha\!=\!17/20\!=\!\alpha_{\xi}\!-\!\gamma/2$",
    "alpha = alpha_xi = 9/10":
        r"$\alpha\!=\!\alpha_{\xi}\!=\!9/10$",
    "alpha = 1":
        r"$\alpha\!=\!1$",
    "alpha free (2 parameters)":
        r"$\alpha$ free (2-parameter)",
}
N_PARAMS = {
    "alpha = 2/3 (old Delta_E theory)":             1,
    "alpha = alpha_xi^2 = 81/100":                   1,
    "alpha = 4/5 = alpha_xi - gamma":               1,
    "alpha = 13/16":                                 1,
    "alpha = D_Omega = beta_pi - gamma = 67/80":    1,
    "alpha = 17/20 = alpha_xi - gamma/2":            1,
    "alpha = alpha_xi = 9/10":                       1,
    "alpha = 1":                                     1,
    "alpha free (2 parameters)":                     2,
}


def main():
    d = json.loads(JSON.read_text(encoding="utf-8"))
    fits = d["family_AICc"]
    fits_sorted = sorted(fits, key=lambda f: f["AICc"])
    best_label = fits_sorted[0]["label"]
    primary_label = "alpha = alpha_xi^2 = 81/100"

    lines = [r"\begin{tabular}{l c c c c c}",
             r"\toprule",
             r"Model & $k$ & $\hat\alpha$ & $R^{2}$ & "
             r"AICc & $\Delta\mathrm{AICc}$ \\",
             r"\midrule"]
    for f in fits_sorted:
        lab = LABEL_TEX[f["label"]]
        k = N_PARAMS[f["label"]]
        if f["label"] == best_label:
            lab = r"\textbf{" + lab + r"} (AICc best)"
        elif f["label"] == primary_label:
            lab = r"\textbf{" + lab + r"} (primary)"
        alpha_cell = f"${f['alpha']:.4f}$"
        r2_cell = f"${f['r2']:.3f}$"
        aicc_cell = f"${f['AICc']:.3f}$"
        d_aicc = f["delta_AICc"]
        if math.isfinite(d_aicc) and abs(d_aicc) < 1e-6:
            daicc_cell = r"$\mathbf{0.000}$"
        else:
            daicc_cell = f"${d_aicc:+.3f}$"
        lines.append(
            f"{lab} & ${k}$ & {alpha_cell} & {r2_cell} & "
            f"{aicc_cell} & {daicc_cell} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
