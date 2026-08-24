# Kabir's Fig 3 kymo runs vs `cluster_env/runs/si_totbiom`

Written 2026-08-20. Question that prompted it: the Fig 3 kymographs show **uneven
final peak spacing**, which has never turned up in the `si_totbiom` runs. What
differs conceptually?

Sources read:

- `Fig 3 Kymo/20260728_fig3kymo_plots.ipynb`, `..._run_fig3kymo.py`,
  `..._fig3kymo_functions.py`, `..._fig3kymo_parameters.md`, `..._run.log`
- `cluster_env/runs/si_totbiom/{base.jl, First look.ipynb}`,
  `.../run1system/base.jl`, `.../run_many/base.jl`,
  `scripts/single_influx.jl:341` (`get_si_sampler_for_paper`)

## What the notebook actually is

Pure re-plotting — no model code, no integration, nothing imported from the
project. It loads `20260728_fig3kymo_data.npz` (recorded biomass fields +
metadata) and draws total-biomass kymographs.

- cells 0–4: generic machinery — a `plot_kymograph` helper, all `p` side by side,
  linear and log time axes
- cells 5–7: hand-tuned figure panels — `p=0.01` alone, `p=1` alone, and the
  two-panel `p ∈ {0.01, 1}` version with `clim=(0,25)`, `ylim=(1e1,1e6)`, which
  looks like the actual Fig 3 panel

Everything conceptually interesting lives in `20260728_run_fig3kymo.py` /
`_functions.py`, documented in `20260728_fig3kymo_parameters.md`.

## What the run behind it was

One community, drawn once, integrated at `p = D_byprod/D_ext ∈ {0.01, 0.1, 1}`,
periodic 1D, `t_max = 1e6`, 1024 cells, only the diffusion coefficients differing
between runs.

## Side-by-side

| | Kabir (Fig 3 Kymo) | `si_totbiom` |
| --- | --- | --- |
| `Ns`, `Nr` | 10, 10 | 20, 20 |
| `prob_eating` | 0.7 | `B/M = 3/20 = 0.15` |
| byproduct leakage `l` | `U(0.3, 0.8)` | `0.0` |
| `l_influx` | 0.999 | 1.0 |
| `num_byproducts` | 2 (fixed) | `Binomial(M, B/M)` |
| `m`, `c`, `cinflux` | lognormal `s=0.3` (~35 % spread) | `base10_lognormal(0, 0.001)` (~0.2 %) |
| survivors | 2 of 10 | many of 20 |
| `D_N` | 1e-6 | 1e-6 |
| domain `L` | `20 · 2π/k*(p)` → 10.92 / 16.21 / 19.32 | fixed 5 for every `p` |
| grid | 1024 (51.2 pts/peak by construction) | 1000 (`run1system`) / 2000 (`run_many`) |
| `t_max` | 1e6, reached in all runs | `T = 1e8` nominal, wall-clock timer stops ~1e5–1e6 |
| IC | multiplicative Gaussian, 1e-3 relative | additive uniform ±1e-3, then `clamp!(·, 0, Inf)` |
| solver | SciPy `BDF`, `rtol=1e-6`, `atol=1e-9`, no positivity | TRBDF2/QNDF, `1e-9`/`1e-9`, `PositiveDomain` |
| acceptance | Turing-unstable at **every** `p`, well-mixed SS by `t=2000` | `maxresid < tol` + non-extinction only |

Linear stability of the accepted community, from the parameter file:

| `p` | `k*` | `λ* = 2π/k*` | max growth rate | `L` |
| --- | --- | --- | --- | --- |
| 0.01 | 11.507 | 0.546 | 0.1281 | 10.921 |
| 0.1 | 7.755 | 0.810 | 0.05295 | 16.205 |
| 1 | 6.504 | 0.966 | 0.009011 | 19.320 |

## The conceptual differences

### 1. Different community ensemble

`get_si_sampler_for_paper` (`scripts/single_influx.jl:341`) gives `l = 0.0` on
byproducts and `m`/`c`/`cinflux` all `base10_lognormal(0, 0.001)` — a strictly
two-layer network (influx → byproducts → dead end) of *near-identical* strains.

Kabir's sampler is a different beast: dense network (`prob_eating = 0.7`),
genuine multi-layer cascade with recycling (`l ~ U(0.3, 0.8)` on byproducts), and
~35 % spread across strains instead of 0.2 %. His community keeps 2 of 10
strains. Almost none of the ecological structure is shared.

### 2. Different acceptance criterion

`sys1` was accepted for `maxresid < tol` and non-extinction only — no linear
stability test at all. Kabir draws communities until one is Turing-unstable at
*every* `p` in the set, including `p = 1`. That is much stronger and rarer, and
it forces the selected community deep into the unstable region at small `p`
(growth rate 0.128 at `p = 0.01` vs 0.009 at `p = 1`).

### 3. Different domain philosophy — the leading suspect

`si_totbiom` fixes `L = 5` for all `p`. Kabir sizes the box per `p` as
`L = 20 · 2π/k*(p)`, so every run starts with exactly 20 fastest-growing
wavelengths of room and `L` grows with `p`.

Two consequences:

- **Peak count.** 20 peaks per box vs whatever `L = 5` buys (if `λ*` is
  comparable, ~5–10, and fewer at large `p`). Coarsening on a periodic domain is
  logarithmically slow in the number of defects: with ~5 peaks you reach the
  1–2-peak end state where "spacing" is trivially uniform; with 20 peaks at
  `t = 1e6` you are still mid-coarsening and frozen defects are the generic
  outcome.
- **Band width.** At `p = 0.01`, growth rate 0.128 is far above threshold, so the
  Eckhaus-stable band of wavenumbers is wide and a whole range of non-uniform
  spacings is *nonlinearly stable* — nothing drives them to equalise. At `p = 1`
  (rate 0.009) it is near-threshold, the band is narrow, and the pattern should
  come out regular. Consistent with the uneven spacing appearing in the
  `p = 0.01` panel specifically.

Note also that Kabir's box is exactly commensurate with `λ*(p)` by construction,
whereas `L = 5` is generically incommensurate with `λ*(p)` and changes its peak
count as `p` varies.

### 4. Initial condition

`perturb_u0_uniform` is *additive* ±1e-3 uniform followed by
`clamp!(·, 0, Inf)`; Kabir's is *multiplicative* Gaussian at 1e-3 relative. Two
real differences: components extinct at the well-mixed fixed point stay
identically zero forever in his runs, whereas in `si_totbiom` they get re-seeded
at ~1e-3 and can come back spatially; and the clamp makes the perturbation of
near-zero components one-sided (nonzero mean).

### 5. Time horizon and recording

`T = 1e8` nominal in `si_totbiom`, but the 3–6 h wall-clock timer in practice
stops the runs around `t ≈ 1e5`–`1e6`; Kabir's hard `t_max = 1e6` was reached in
all three runs. Comparable in practice. His recording grid is a merged
linear+log grid of 599 rows chosen for the kymograph; `si_totbiom` uses
`make_stepped_saver_callback` at a fixed step.

### 6. Numerics

SciPy `BDF` at `rtol = 1e-6`, `atol = 1e-9`, no positivity enforcement, vs
TRBDF2/QNDF at `1e-9`/`1e-9` with `PositiveDomain`. A 1e-6 relative tolerance
over `t = 1e6`, on a system whose slow modes are marginal defect motion, is loose
enough to be worth ruling out before treating the uneven spacing as purely
physical — but probably not the main story.

## Conclusion

The two most likely causes of the uneven spacing are **(3)** — a 20-peak box far
above threshold, where disordered spacing is genuinely nonlinearly stable and
coarsening has not finished by `t = 1e6` — and **(1)/(2)**, a much more
heterogeneous, recycling community selected specifically for broad-band
instability. The `L = 5` `si_totbiom` runs are a different regime: few peaks,
near-identical strains, no secondary leakage.

Cheapest test: rerun one `si_totbiom` system with `L` set to ~20 `λ*` instead of
5 and see whether uneven spacing appears at small `p`.
