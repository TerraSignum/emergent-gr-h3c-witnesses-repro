"""Helper: locate per-regime D1 NPZ files for the Galerkin verification
scripts.

Search order:
  (1) bundled location: <repo>/data/d1_runs/d1_p<X>.npz
  (2) external location: <repo-parent>/d1_lattice_payload/d1_p<X>.npz
                         <repo-parent>/d1_lattice_payload/p<X>/d1_p<X>.npz
                         <repo-parent>/results_d1_p5n64/d1_p5n64.npz
                         <repo-parent>/results_d1_p5n100/d1_p5n100.npz

The bundled path is checked first to support standalone reproduction
when the user has placed the NPZ files into <repo>/data/d1_runs/. If
not found, the external location (relative to the parent directory of
the repo) is used, matching the developer's working directory layout.

If neither path resolves, returns None so the caller can degrade
gracefully (e.g. fall back to the bundled JSON certificate).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional


def find_d1_npz(regime: str, repo_root: Path) -> Optional[Path]:
    """Return Path to the requested D1 NPZ or None if not found.

    Args:
      regime: regime tag, e.g. "P0", "P5", "P5N64", "P5N100"
      repo_root: Path to the emergent-gr-closure-repro root

    Returns:
      Path or None.
    """
    regime = regime.lower()
    bundled_root = repo_root / "data" / "d1_runs"
    external_root = repo_root.parent

    # Build candidate list.
    candidates: list[Path] = []

    # Bundled location (primary).
    candidates.append(bundled_root / f"d1_{regime}.npz")

    # External developer paths.
    if regime in ("p0", "p1", "p2prime", "p3", "p4", "p5"):
        candidates.append(external_root / "d1_lattice_payload"
                          / f"d1_{regime}.npz")
    if regime in ("p6", "p7", "p8"):
        candidates.append(external_root / "d1_lattice_payload"
                          / regime / f"d1_{regime}.npz")
    if regime in ("p5n64",):
        candidates.append(external_root / "results_d1_p5n64"
                          / "d1_p5n64.npz")
    if regime in ("p5n100",):
        candidates.append(external_root / "results_d1_p5n100"
                          / "d1_p5n100.npz")

    for p in candidates:
        if p.exists():
            return p
    return None


def standalone_message(regime: str) -> str:
    """Helpful message when NPZ not found."""
    return (
        f"D1 NPZ for regime {regime} not found in either bundled "
        f"<repo>/data/d1_runs/ or external <repo-parent>/results_d1_*. "
        f"This script requires the per-node lattice state. Either:\n"
        f"  (a) populate <repo>/data/d1_runs/ with the bundled NPZ "
        f"      files (download instructions in README); or\n"
        f"  (b) verify the recomputed result against the frozen "
        f"      JSON certificate at <repo>/data/galerkin_*.json."
    )
