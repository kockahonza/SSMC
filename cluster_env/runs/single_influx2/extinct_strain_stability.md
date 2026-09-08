# Extinct strains in spatial instabilities

Scan of all 25,000 systems in `gd1_260731_115325.jld2` (5 leakages x 50 K x 100 repeats,
`DN = 0`, `DR = 1`). For each of the 6,605 systems with a positive peak in its dispersion
relation: find the peak over `lsks`, take the eigenvector of the leading eigenvalue there,
normalize by its largest component, and measure the largest component sitting on a strain with
`|N_i| < 1e-10` in the homogeneous steady state. See
`Extinct strains in unstable modes.ipynb` for the code.

## Results

- **6,375 clean spatial instabilities** (`lscode = 2`, `k0mrl < 0`): the largest extinct-strain
  component of the peak eigenvector is **<= 2.0e-9**, typically ~1e-12. That is numerical
  residue - it tracks the leftover biomass, not physics.
- **196 rows pass a 1e-3 relative threshold**, all of them `lscode = 23`. They are **not spatial
  instabilities**: in every one, `k0mrl == maxmrl ==` the extinct strain's invasion eigenvalue,
  and the dispersion relation is flat - max minus min over the whole k range is ~1e-10 against
  peaks of 1e-3 to 0.8. The `argmax` lands on a large k only because eigensolver noise grows with
  `k^2 D`. These are systems where the ODE steady state is invadable by a strain the solver left
  at 1e-20 to 1e-150, so it can never numerically grow back.
- Across the clean unstable systems, the smallest steady-state biomass of a strain carrying more
  than 0.1 of the peak eigenvector is ~5e-3 (measured on a 1-in-6 subsample), eight orders of
  magnitude above the extinction threshold.

## Why it can't happen here

With `N_i = 0` the extinct strain's Jacobian row decouples: `M1[i, j] = 0` for `j != i` and
`M1[i, Ns+a] ∝ N_i`. So

- its eigenvalue is exactly `M1[i, i]`, and since `DN = 0` that branch is **flat in k** - it can
  only be the peak if it is already the max at k = 0, ie `k0mrl >= 0`, the messy lscodes;
- for any other eigenvalue `λ`, `v_i = (Σ_a M[i, Ns+a] v_{Ns+a}) / (λ - λ_i) ∝ N_i` - the
  component is proportional to the extinct biomass.

Participation in the mode requires real abundance.

## Caveat

This only rules out extinct strains participating in the *linear* instability of the homogeneous
steady state. Whether a pattern, once formed, creates niches that let an extinct strain re-invade
is a different calculation (stability of the patterned state) that this dataset cannot answer.
