# emergent-gr-h3c-witnesses-repro

[![CI: reproduce](https://github.com/TerraSignum/emergent-gr-h3c-witnesses-repro/actions/workflows/reproduce.yml/badge.svg)](https://github.com/TerraSignum/emergent-gr-h3c-witnesses-repro/actions/workflows/reproduce.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Reproducibility package for the **multi-observable indirect-witness
chain** that bounds the Einstein-identity gap from above on the
relational emergent-gravity construction. Topic-focused companion
repository to `emergent-gr-closure-repro` (Paper 4) which reports
the per-node 4×4 Galerkin closure as the principal direct test.

## What this repository contains

The framework's universal-Richardson bound on the Einstein-identity
gap is supplemented by a multi-observable chain of indirect
witnesses. This repository bundles four such witnesses with their
reproducibility certificates:

1. **Chirality-balance five-point fit:** a log-log fit of the
   chirality-deviation observable across $N \in
   \{28,30,36,42,50\}$ recovers the universal exponent
   $\alpha_{\text{gap}} = 0.6355$ within 4.7% of the analytical
   $2/3$ target ($R^{2} = 0.78$).

2. **Ricci-scalar nine-point witness:** the per-node mean
   $\bar R$ across the canonical lattice ladder traces a power-law
   decay with $\alpha = 2/3$ at $R^{2} = 0.83$, complementary to
   the rigorous Einstein-identity gap.

3. **Off-diagonal spectral-stress multi-N:** the off-diagonal
   spatial stress component decays with $\alpha = 2.54$ on the
   bundled extension lattice ladder; the bound
   $\|T_{\mu\nu}^{\text{off}}\|_{F}(N) \to 0$ is verified
   structurally.

4. **Look-Elsewhere Bonferroni multiplicity correction:** the
   joint LEE band on the four-component $\Lambda_{\mu\nu}$
   identification under independent regressor choices is
   reported with explicit Bonferroni correction.

## Tier-1 reproduction (frozen JSON, no compute)

```bash
pip install -e .
pytest tests/
```

Expected: 13 unit tests pass.

## Tier-2 reproduction (D1 lattice run, optional)

The off-diagonal stress and Ricci-scalar witnesses at higher
resolution require the D1 lattice NPZ files which are not bundled.

## Companion paper structure

This repo extracts the indirect-witness chain from the parent
emergent-gravity closure paper (P4) and presents it as a focused
diagnostic-companion paper. The parent retains the per-node 4×4
Galerkin closure as the principal direct test of the
Einstein-identity gap.

## License

MIT.