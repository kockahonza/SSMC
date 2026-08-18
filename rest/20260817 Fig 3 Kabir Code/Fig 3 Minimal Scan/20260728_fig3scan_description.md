# Fig. 3 Minimal Scan — byproduct mobility scan

Documentation of the calculation implemented in

* `20260727_fig3scan_functions.py` — model, linear stability, solver, peak counting, plotting
* `20260727_run_fig3scan.py` — driver: task farm, checkpointing, figures
* `20260727_run_fig3scan.sbatch` — cluster submission
* `20260727_fig3scan_peaks.npz` — saved results
* `slurm-47863.out` — log of the run that produced the saved results

All line references below are to those files as they stand.

---

## 1. Overview

This scan asks a single question about the three-variable ("minimal") reduction of the
Modified MiCRM: **how does the number of spatial peaks in a Turing pattern evolve in time,
and how does that evolution depend on the mobility of the metabolic byproduct relative to
the externally supplied resource, $p = D_x/D_{\rm ext}$?** Each run starts from the
homogeneous coexistence fixed point in a periodic 1D box that has been sized, using the
linear dispersion relation, to hold exactly 100 copies of the fastest-growing wavelength
$\lambda^* = 2\pi/k^*$. Every run therefore *begins* pattern formation with ~100 peaks by
construction, so any subsequent drop in the peak count is unambiguously **coarsening**
rather than a difference in domain size. Running six values of $p$ times ten independent
noise seeds out to $t = 10^8$ (about seven decades past pattern formation) shows that
coarsening is essentially absent for slow byproducts ($p \lesssim 0.1$, count stays at the
linear-instability value) and becomes progressively stronger and faster as the byproduct
becomes as mobile as, or more mobile than, the external resource ($p \gtrsim 0.5$), where
the pattern coarsens by more than an order of magnitude and shows no sign of arresting by
$t_{\max}$.

---

## 2. Model equations

### Governing PDEs

One microbial strain of biomass density $n$ consumes an externally supplied resource
$r_{\rm ext}$ and the byproduct $r_x$ that its own metabolism excretes. In one spatial
dimension $x \in [0, L)$ with periodic boundaries (docstring
`20260727_fig3scan_functions.py:8-10`; implementation
`20260727_fig3scan_functions.py:227-231` and `:240-241`):

$$
\frac{\partial n}{\partial t}
= n\Big[\, c_{\rm ext}(1-l)\, r_{\rm ext} + c_x\, r_x - m \,\Big]
\;+\; D_n \frac{\partial^2 n}{\partial x^2}
$$

$$
\frac{\partial r_{\rm ext}}{\partial t}
= K - \kappa\, r_{\rm ext} - c_{\rm ext}\, r_{\rm ext}\, n
\;+\; D_{\rm ext} \frac{\partial^2 r_{\rm ext}}{\partial x^2}
$$

$$
\frac{\partial r_x}{\partial t}
= c_{\rm ext}\, l\, r_{\rm ext}\, n - \kappa\, r_x - c_x\, r_x\, n
\;+\; D_x \frac{\partial^2 r_x}{\partial x^2}
$$

The state vector is ordered $(n, r_{\rm ext}, r_x)$ throughout
(`20260727_fig3scan_functions.py:107-108`).

**Note on the parameter values actually used.** With the defaults in this folder the
leakage is total, $l = 1$, so the direct growth contribution of the external resource
vanishes: $c_{\rm ext}(1-l) = 0$. All biomass is fed through the byproduct, and the
biomass equation reduces in practice to
$\partial_t n = n\,(c_x r_x - m) + D_n\,\partial_x^2 n$. The external resource is a pure
substrate that is converted one-for-one into byproduct. This is an intentional choice
(comment at `20260727_fig3scan_functions.py:46-51`), not an accident, but it means the
$c_{\rm ext}(1-l)$ terms are dormant in every number reported here.

### Noise

**There is no noise term in the dynamics.** The PDE integrated is fully deterministic;
stochasticity enters *only* through the initial condition. This is worth stating plainly
because the driver refers to "noise seeds" and the figures to "replicates" — the
replicates differ solely in their seeded initial perturbation, not in any ongoing
fluctuation.

### Boundary and initial conditions

*Boundary conditions:* periodic in $x$, imposed by the wrap-around of `np.roll` in the
Laplacian stencil (`20260727_fig3scan_functions.py:240`) and mirrored in the corner
entries of the Jacobian sparsity pattern (`:257-259`).

*Initial condition:* the homogeneous coexistence fixed point $u^*$ under a small
independent multiplicative Gaussian perturbation, one draw per component per grid cell
(`20260727_fig3scan_functions.py:408-409`):

$$
u_i(x_j, t{=}0) \;=\; u_i^*\,\big[\,1 + \varepsilon\, \xi_{ij}\,\big],
\qquad \xi_{ij} \sim \mathcal{N}(0,1)\ \text{i.i.d.},\qquad \varepsilon = 10^{-3}.
$$

The fixed point is obtained in closed form (`20260727_fig3scan_functions.py:116-145`) by
eliminating $r_{\rm ext} = K/(\kappa + c_{\rm ext} n)$ and
$r_x = c_{\rm ext} l\, r_{\rm ext} n / (\kappa + c_x n)$, which reduces the growth balance
to a quadratic in biomass (`:131-133`):

$$
m\,c_{\rm ext} c_x\, n^2
+ \big[\,m\,\kappa\,(c_{\rm ext}+c_x) - K c_{\rm ext} c_x\,\big] n
+ m\,\kappa^2 - K c_{\rm ext} \kappa (1-l) \;=\; 0 ,
$$

of whose two roots the **larger** is taken (`:142`); the smaller is a saddle. A negative
discriminant raises `ValueError` — for these parameters the pair is born in a saddle-node
at $K = 4$. With the values used ($K=5$), the fixed point is

$$
(n^*, r_{\rm ext}^*, r_x^*) = (2.6180,\; 1.3820,\; 1.0000)
$$

independent of $p$ (verified numerically; $r_x^* = m/c_x = 1$ exactly, since $l=1$ forces
$c_x r_x^* = m$).

### Linear stability / dispersion relation

A perturbation $\propto e^{ikx}$ of the homogeneous state grows at the largest real
eigenvalue of $J - k^2 \operatorname{diag}(D_n, D_{\rm ext}, D_x)$
(`20260727_fig3scan_functions.py:207-208`):

$$
\sigma(k) \;=\; \max_j \operatorname{Re}\,
\lambda_j\!\left[\, J(u^*) - k^2 \operatorname{diag}(D_n, D_{\rm ext}, D_x) \,\right],
$$

with the analytic well-mixed Jacobian (`20260727_fig3scan_functions.py:156-161`)

$$
J = \begin{pmatrix}
c_{\rm ext}(1-l) r_{\rm ext} + c_x r_x - m & n\,c_{\rm ext}(1-l) & n\,c_x \\[2pt]
-c_{\rm ext} r_{\rm ext} & -\kappa - c_{\rm ext} n & 0 \\[2pt]
c_{\rm ext} l\, r_{\rm ext} - c_x r_x & c_{\rm ext} l\, n & -\kappa - c_x n
\end{pmatrix}_{u = u^*}.
$$

Then $k^* = \arg\max_k \sigma(k)$, $\sigma^* = \sigma(k^*)$, $\sigma_0 = \sigma(0)$, and
$\lambda^* = 2\pi/k^*$ (`:181-188`, `:210-212`). A Turing instability is declared when
$\sigma_0 < 0 < \sigma^*$ (`:182-184`). This holds for all six $p$ scanned; numerically
$\sigma_0 = -1$ exactly at every $p$.

The box length is then set by $k^*$ (`20260727_fig3scan_functions.py:393`):

$$
L \;=\; N_{\rm peaks}\,\lambda^* \;=\; N_{\rm peaks}\,\frac{2\pi}{k^*},
\qquad N_{\rm peaks} = 100 .
$$

### Symbol table

| Symbol | Code variable | Meaning |
|---|---|---|
| $n$ | `u[0]`, `n`; `biomass` after solve | strain biomass density |
| $r_{\rm ext}$ | `u[1]`, `r_ext` | externally supplied resource concentration |
| $r_x$ | `u[2]`, `r_x` | metabolic byproduct concentration |
| $c_{\rm ext}$ | `prm.c_ext` / `C_EXT` | consumption rate of the external resource |
| $c_x$ | `prm.c_x` / `C_X` | consumption rate of the byproduct |
| $l$ | `prm.l` / `LEAKAGE` | fraction of consumed external resource leaked as byproduct |
| $m$ | `prm.m` / `DEATH` | strain death rate |
| $K$ | `prm.K` / `K_INFLUX`, `args.K` | external supply (influx) of $r_{\rm ext}$ |
| $\kappa$ | `prm.kappa` / `KAPPA` | resource dilution/decay rate (both resources) |
| $D_n$ | `prm.D_n` / `D_N` | biomass diffusivity |
| $D_{\rm ext}$ | `prm.D_ext` / `D_EXT` | external-resource diffusivity (sets the mobility unit) |
| $D_x$ | `prm.D_x` | byproduct diffusivity, $= p\,D_{\rm ext}$ (`from_p`, `:100-103`) |
| $p$ | `p`, `args.ps`, `P_VALUES` | **scanned** byproduct mobility ratio $D_x/D_{\rm ext}$ |
| $\varepsilon$ | `NOISE_AMPLITUDE` | amplitude of the multiplicative seeding perturbation |
| $\xi_{ij}$ | `rng.standard_normal((3, ssize))` | i.i.d. standard normals |
| $u^*$ | `u_ss` = `steady_state(prm)` | homogeneous coexistence fixed point |
| $J$ | `jacobian(prm, u)` | well-mixed $3\times3$ Jacobian |
| $\sigma(k)$ | `growth` inside `dispersion` | growth rate of mode $k$ |
| $k^*$ | `disp.k_star`, npz `k_star` | fastest-growing wavenumber |
| $\sigma^*$ | `disp.max_growth`, npz `max_growth` | its growth rate |
| $\sigma_0$ | `disp.growth_0` | growth rate of the $k=0$ mode (not saved) |
| $\lambda^*$ | `disp.wavelength` | $2\pi/k^*$ |
| $L$ | `L_box`, npz `L_box` | domain length |
| $N_{\rm peaks}$ | `n_peaks`, npz `n_peaks` | wavelengths the box is sized to hold |
| $\Delta x$ | `prm.dx` | grid spacing, $L/\texttt{ssize}$ |
| $N_x$ | `ssize`, npz `ssize` | number of grid cells |

**Units.** Everything is dimensionless. Because $\kappa = m = c_{\rm ext} = c_x = 1$, time
is measured in units of the death/dilution time and length in units of
$\sqrt{D_{\rm ext}/\kappa} = 1$; concentrations are in units set by $K/\kappa$. The code
never attaches physical units.

---

## 3. Numerical method

**Spatial discretisation.** Uniform grid of $N_x = 2048$ cells on a periodic domain of
length $L$, so $\Delta x = L/N_x$ (`20260727_fig3scan_functions.py:394`). The Laplacian is
the standard second-order three-point central difference with periodic wrap
(`:240-241`):

$$
\left.\frac{\partial^2 u_i}{\partial x^2}\right|_j
\approx \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{\Delta x^2},
\qquad j \pm 1 \ \text{taken mod } N_x .
$$

Because $L \propto 1/k^*$ and $N_x$ is fixed, the resolution per wavelength is the same at
every $p$: $\lambda^*/\Delta x = N_x/N_{\rm peaks} = 20.48$ points per wavelength
(confirmed numerically for all six $p$). Absolute $\Delta x$ ranges from $0.0617$ ($p=0.05$,
$L=126.38$) to $0.0964$ ($p=2$, $L=197.49$).

**Time stepping.** Method-of-lines: the $3 N_x = 6144$ ODEs are handed to
`scipy.integrate.solve_ivp` with `method="BDF"` — an adaptive, variable-order implicit
backward-differentiation scheme — with `rtol=1e-6`, `atol=1e-9`
(`20260727_fig3scan_functions.py:262-276`). **There is no user-set timestep**; the step is
chosen adaptively by the solver, which is what makes seven decades of time tractable. The
Jacobian is not supplied analytically; instead its *sparsity pattern* is passed via
`jac_sparsity`, built as the Kronecker product of a dense $3\times3$ reaction block with a
periodic tridiagonal stencil in the cell index (`:244-259`), so the solver estimates and
factorises it sparsely.

**Recording grid.** The solution is only ever evaluated at 300 logarithmically spaced
times spanning $t \in [0.1,\ 10^8]$ (`log_times`, `:279-287`; called at `:411`):
`np.logspace(log10(min(0.1, t_max/1e3)), log10(t_max), 300)`. Successive times differ by a
factor $1.0718$, i.e. $\approx 37.4$ samples per decade. Verified from the npz:
`t[0] = 0.1`, `t[-1] = 1e8`, `t.size = 300`.

**Integration / burn-in duration.** There is no separate burn-in: integration starts at
$t=0$ from the perturbed fixed point (the initial time passed to the solver is
`t_eval[0] = 0.1`, so the first recorded sample is essentially the initial condition) and
runs to $t_{\max} = 10^8$. Pattern formation (first non-zero peak count) occurs at
$t \approx 34$ for $p=0.05$ up to $t \approx 1.5\text{–}1.8\times10^3$ for $p=2$, in
inverse proportion to $\sigma^*$; the remaining five to six decades are coarsening.

**RNG seeding.** A single master seed `--seed 20260727` is expanded by
`np.random.SeedSequence(...).generate_state(n_p * n_rep)` into one independent `uint32` per
$(p,\text{replicate})$ slot, reshaped to $(6,10)$ (`20260727_run_fig3scan.py:251-252`).
Each worker builds `np.random.default_rng(seed)` (PCG64) from its own slot's value
(`20260727_fig3scan_functions.py:408`). All 60 seeds are saved in the npz, so any single
run is independently reproducible. Note that the same set of 10 seeds is *not* reused
across $p$ — every $(p, \text{rep})$ pair has a distinct seed, so replicate index carries
no meaning across $p$.

**Replicate structure and averaging.** $6$ values of $p$ $\times$ $10$ replicates $= 60$
fully independent solves, no coupling between them. There is no averaging inside a run.
Averaging happens only at plot time: the replicate mean and min–max envelope of the peak
count are taken at each recorded time, over the replicates that have a pattern at that
time (see §5). Runs are farmed out one per core with `multiprocessing.Pool.imap_unordered`
and BLAS/OpenMP capped to one thread per worker (`20260727_run_fig3scan.py:37-41`,
`:274-281`). The npz is rewritten after every completed run, so the file on disk is always
a valid partial snapshot.

**Wavenumber grid for the dispersion relation.** $\sigma(k)$ is sampled on
`KS = np.linspace(1e-3, 60.0, 6000)` (`20260727_fig3scan_functions.py:66`), i.e. spacing
$\approx 0.01$. $k^*$ (and hence $L$) is therefore quantised at the $10^{-2}$ level — a
$\lesssim 0.3\%$ effect on $L$, harmless here but worth knowing.

---

## 4. Parameters

### Fixed parameters

| Symbol | Code variable | Value | Meaning / units |
|---|---|---|---|
| $c_{\rm ext}$ | `C_EXT` (`functions.py:52`) | $1.0$ | external-resource consumption rate (per biomass per time) |
| $c_x$ | `C_X` (`functions.py:53`) | $1.0$ | byproduct consumption rate |
| $l$ | `LEAKAGE` (`functions.py:54`) | $1.0$ | leaked fraction (dimensionless); $l=1 \Rightarrow$ total leakage |
| $\kappa$ | `KAPPA` (`functions.py:55`) | $1.0$ | resource dilution rate (per time) |
| $m$ | `DEATH` (`functions.py:56`) | $1.0$ | strain death rate (per time) |
| $K$ | `K_INFLUX` (`functions.py:57`), `--K` | $5.0$ | external supply of $r_{\rm ext}$ (concentration per time) |
| $D_n$ | `D_N` (`functions.py:61`) | $10^{-3}$ | biomass diffusivity — strain nearly immotile |
| $D_{\rm ext}$ | `D_EXT` (`functions.py:62`) | $1.0$ | external-resource diffusivity; sets the mobility unit |
| $\varepsilon$ | `NOISE_AMPLITUDE` (`functions.py:69`) | $10^{-3}$ | multiplicative seeding-noise amplitude |
| $N_{\rm peaks}$ | `--n-peaks` (`run:220`) | $100$ | wavelengths per box (sets $L$) |
| $N_x$ | `--ssize` (`run:217`) | $2048$ | grid cells $\Rightarrow$ 20.48 points/wavelength |
| $t_{\max}$ | `--t-max` (`run:215`) | $10^8$ | final time |
| $n_{\rm times}$ | `--n-times` (`run:222`) | $300$ | log-spaced recording times |
| $t_{\min}$ | `log_times` default (`functions.py:279`) | $0.1$ | first recorded time |
| — | `rtol`, `atol` (`functions.py:263`) | $10^{-6}$, $10^{-9}$ | BDF tolerances |
| — | `--replicates` (`run:213`) | $10$ | noise seeds per $p$ |
| — | `--seed` (`run:226`) | $20260727$ | master seed |
| — | `KS` (`functions.py:66`) | `linspace(1e-3, 60, 6000)` | wavenumber grid for $\sigma(k)$ |
| — | `min_contrast` (`functions.py:295`) | $0.05$ | peak-detection contrast gate (§5) |
| — | `prominence_frac` (`functions.py:296`) | $0.05$ | peak-detection prominence gate (§5) |

$K = 5$ is described in the code as the one genuinely free parameter: the coexistence fixed
point is born in a saddle-node at $K=4$ and the Turing instability is said to disappear
again near $K=8$, so $K=5$ sits mid-window (comment `functions.py:46-51`). The $K=8$ upper
edge is a claim in the comment, not something this code computes.

### Scanned parameter

| Symbol | Code variable | Values | Spacing |
|---|---|---|---|
| $p = D_x/D_{\rm ext}$ | `P_VALUES` (`run:71`), `--ps` (`run:211`), sbatch `:40` | $0.05,\ 0.1,\ 0.25,\ 0.5,\ 1.0,\ 2.0$ | **6 points, hand-chosen quasi-logarithmic** (ratios $2,\,2.5,\,2,\,2,\,2$) — not generated by `logspace` |

$p$ is the *only* scanned parameter. Since $D_{\rm ext} = 1$, $p$ is numerically $D_x$. The
range endpoints are motivated in the driver comment (`run:69-70`): below $p \approx 0.05$
the peak count stays locked at its linear-instability value, and above $p \approx 2.2$ the
instability itself disappears. Neither bound is computed by this script.

Quantities *derived* from $p$ (one value per $p$, saved in the npz, no replicate axis):

| $p$ | $k^*$ | $\lambda^*$ | $\sigma^*$ | $L$ | $\Delta x$ |
|---|---|---|---|---|---|
| 0.05 | 4.9717 | 1.2638 | $+0.10966$ | 126.378 | 0.0617 |
| 0.1  | 4.5117 | 1.3926 | $+0.08529$ | 139.265 | 0.0680 |
| 0.25 | 3.9916 | 1.5741 | $+0.05160$ | 157.410 | 0.0769 |
| 0.5  | 3.6816 | 1.7067 | $+0.02955$ | 170.667 | 0.0833 |
| 1.0  | 3.4215 | 1.8364 | $+0.01323$ | 183.638 | 0.0897 |
| 2.0  | 3.1815 | 1.9749 | $+0.00290$ | 197.493 | 0.0964 |

More mobile byproduct $\Rightarrow$ longer wavelength and much weaker instability
($\sigma^*$ falls by a factor $\sim 38$ across the range).

---

## 5. Observables and post-processing

Exactly one observable is computed: **the number of peaks in the biomass field**, as a
function of time.

### Peak counting

`count_peaks` (`20260727_fig3scan_functions.py:295-320`) is applied to the **biomass row
only** — the resource fields are never counted (`:414`, `sol.y[:ssize].T`). For a 1D
periodic field $f$ with $f_{\min}, f_{\max}, \bar f$ its min, max and mean:

1. **Contrast gate** (`:315`). If $\bar f \le 0$ or
   $$\frac{f_{\max} - f_{\min}}{\bar f} < \texttt{min\_contrast} = 0.05,$$
   the count is **0**. This is essential: before the instability has grown out of the
   $10^{-3}$ seed noise, the field is flat and its local maxima are pure noise — naive
   counting returns 100–200 spurious peaks. With the gate, a zero is a *real measurement*
   meaning "no pattern yet", and the first time a trace leaves zero is the
   pattern-formation time.
2. **Periodic peak finding** (`:319-320`). The field is tiled three times,
   `scipy.signal.find_peaks` is run on the triple with
   $\texttt{prominence} = \texttt{prominence\_frac}\times(f_{\max}-f_{\min}) = 0.05\,(f_{\max}-f_{\min})$,
   and only maxima whose index falls in the middle copy are counted. Tiling handles
   wrap-around exactly; the prominence threshold discards small ripples riding on a formed
   pattern. Note the prominence threshold is computed from the range of a *single* copy,
   while `find_peaks` operates on the tiled array — the two are consistent since tiling
   does not change the range.

`peaks_over_time` (`:323-325`) applies this to every recorded time, giving an integer
`(n_times,)` trace. No spectra, structure factors, or other normalisations are computed
anywhere; the spatial profiles are discarded as soon as they are counted, which is why the
whole scan fits in a 150 kB npz.

### Reading convention: zero vs NaN

`formed_peaks` (`:328-341`) is the single documented convention for reading the saved
`peaks` array, and it is applied before every statistic and every plot:

$$
\texttt{formed\_peaks}(x) = \begin{cases} x & x > 0 \\ \text{NaN} & \text{otherwise} \end{cases}
$$

so that **NaN** (never run) and **0** (run, but field still too flat to hold a pattern) are
both excluded from means and envelopes. Averaging a zero in as "a count of zero peaks"
would report coarsening that did not happen. The corollary, stated in the docstring: to ask
*when* patterns formed you must use the raw saved array, because the zeros carry that
information and `formed_peaks` throws it away.

### Plot-time statistics

In `plot_peaks_vs_time` (`:435-474`): for each $p$, at each time where **at least one**
replicate has a pattern, the shaded band is `nanmin`–`nanmax` over replicates and the solid
line is `nanmean` over replicates. Times where no replicate has formed a pattern are
dropped from the curve entirely — hence the curves start at different $t$ for different
$p$, and the band shows spread in *count*, not spread in *formation time*. In
`plot_final_peaks` (`:477-506`) the same masking is applied to the last column
`peaks[:, :, -1]`; individual replicates are drawn as faint points and the mean as a line.
Both figures are drawn on log–log axes.

### Contents of `20260727_fig3scan_peaks.npz`

Written by `save_npz` (`20260727_run_fig3scan.py:91-130`); verified by loading the file.
Here $n_p = 6$, $n_{\rm rep} = 10$, $n_{\rm times} = 300$.

| Key | Shape | dtype | Contents |
|---|---|---|---|
| `p` | `(6,)` | float64 | the mobilities scanned: `[0.05 0.1 0.25 0.5 1. 2.]` |
| `seeds` | `(6, 10)` | uint32 | the RNG seed of every $(p,\text{rep})$ slot |
| `t` | `(300,)` | float64 | recording grid, $0.1 \to 10^8$, log-spaced |
| `peaks` | `(6, 10, 300)` | float64 | **the result.** Peak count; NaN = never run, 0 = run but no pattern yet. In this file: 0 NaNs — all 60 slots ran. |
| `done` | `(6, 10)` | bool | slot was run at all. All 60 True. |
| `success` | `(6, 10)` | bool | solver reached $t_{\max}$. All 60 True. |
| `k_star` | `(6,)` | float64 | $k^*$ per $p$ (no replicate axis — follows from $p$ alone) |
| `max_growth` | `(6,)` | float64 | $\sigma^*$ per $p$ |
| `L_box` | `(6,)` | float64 | $L$ per $p$ |
| `t_max` | `()` | float64 | $10^8$ |
| `ssize` | `()` | int64 | $2048$ |
| `n_peaks` | `()` | int64 | $100$ |
| `K` | `()` | float64 | $5.0$ |

Note that `growth_0` ($\sigma_0$), the fixed-point values, and the seeds' derivation are
not stored (the first two are trivially recomputable from `p` and `K`).

Final counts at $t_{\max}$ (mean over 10 replicates, [min, max]; from `slurm-47863.out`
and reproduced from the npz):

| $p$ | peaks at $t_{\max}$ | fraction of the initial 100 |
|---|---|---|
| 0.05 | 99.9 [97, 101] | ~1.00 |
| 0.1  | 96.1 [93, 98]  | ~0.96 |
| 0.25 | 85.8 [84, 87]  | ~0.86 |
| 0.5  | 62.7 [60, 66]  | ~0.63 |
| 1.0  | 9.3 [8, 10]    | ~0.09 |
| 2.0  | 6.5 [6, 7]     | ~0.065 |

(Counts slightly above 100 immediately after pattern formation — up to 111 at $p=0.05$ —
are the detector picking up a mode marginally shorter than $\lambda^*$ plus residual
noise structure, not a physical excess; they settle within a decade.)

---

## 6. Figures

Both are drawn by `draw_figures` (`20260727_run_fig3scan.py:140-157`) and can be redrawn
from the npz alone with `python 20260727_run_fig3scan.py --plot-only`.

### 6.1 `20260727_fig3scan_peaks.png` / `.svg` — coarsening trajectories

Single panel, log–log. **x-axis:** time $t$, from $\sim 30$ to $10^8$. **y-axis:** number
of biomass peaks, $\sim 5$ to $\sim 110$. One coloured band per $p$ (viridis, dark violet
$= p{=}0.05$ through yellow $= p{=}2$, coloured in log-$p$ via `p_shades`, `:427-432`):
the solid line is the replicate mean, the translucent envelope the replicate min–max. A
horizontal grey dashed line marks $N_{\rm peaks} = 100$. Title records the replicate count
and $t_{\max}$; legend in two columns.

What to read off it:

* Every curve enters at ~100 peaks, confirming that the box sizing worked — all six $p$
  start from the same pattern by construction.
* Entry time increases with $p$ ($t \approx 34$ at $p=0.05$ to $t \approx 1.6\times10^3$ at
  $p=2$), tracking $1/\sigma^*$.
* $p = 0.05$ and $p = 0.1$ are essentially **flat**: the linear-instability wavelength is
  the final state, no coarsening over seven decades.
* $p = 0.25$ and $p = 0.5$ show a single early drop (around $t \sim 10^3$) to a new plateau
  (~86 and ~63 peaks) and then arrest.
* $p = 1$ and $p = 2$ never arrest: after a sharp initial collapse they continue to coarsen
  as a slow, roughly log-in-time staircase all the way to $t = 10^8$, reaching ~9 and ~6.5
  peaks and still falling. The stepped appearance is the discrete nature of the count
  (individual peak-merger events), and the replicate band is widest here because merger
  times are the most noise-sensitive.
* Conclusion: byproduct mobility is the control parameter for whether the Turing pattern is
  **wavelength-selecting** (small $p$: the linearly selected wavelength survives) or
  **coarsening** (large $p$: the pattern is only transiently at $\lambda^*$ and drifts
  towards a domain-scale state).

### 6.2 `20260727_fig3scan_final.png` / `.svg` — final count vs mobility

Single panel, log–log. **x-axis:** $p = D_x/D_{\rm ext}$, $0.05$ to $2$. **y-axis:** number
of peaks at $t = 10^8$, $\sim 5$ to $\sim 100$. Green (`(27,158,119)/255`) filled circles
joined by a line = replicate mean; faint translucent circles = the 10 individual
replicates at each $p$; grey dashed horizontal line at 100 labelled "linear instability
($L/\lambda^*$)".

What to read off it:

* A monotone decrease of the surviving peak count with $p$, nearly flat for
  $p \le 0.25$ and then falling off a cliff: a factor $\sim7$ drop between $p = 0.5$ and
  $p = 1$, i.e. the transition from arrested to runaway coarsening is sharp and sits near
  $p \approx 1$ — where the byproduct and the external resource are equally mobile.
* Replicate scatter is tight everywhere (visible only at $p = 1$ and $p = 2$), so the trend
  is not a noise artefact.
* **Important caveat on the dashed reference line.** It is labelled "linear instability
  ($L/\lambda^*$)" but is drawn at `n_peaks_target = 100` (`:498-500`), and $L$ was
  *defined* as $100\lambda^*$. The line is therefore true by construction, not an
  independent prediction — it is the "no coarsening" baseline against which the measured
  counts are compared, and should be read that way.
* Because $t_{\max}$ is finite and the $p \ge 1$ traces have not arrested, the two
  right-most points are **not** asymptotic values; they are "how far it got by $t=10^8$".

---

## 7. Compute cost

**Requested** (`20260727_run_fig3scan.sbatch:2-7`): partition `BIOP`, walltime limit
`72:00:00`, one node (`-N1`), one task (`-n1`), `--cpus-per-task=36`, job name `fig3scan`.
BLAS/OpenMP threading is pinned to 1 in both the sbatch file (`:32-37`) and the Python
driver before numpy import (`20260727_run_fig3scan.py:37-41`).

**Parallelism.** Each of the 60 runs is one independent single-threaded stiff solve;
parallelism is coarse-grained, one run per core, via `multiprocessing.Pool` sized to
`len(os.sched_getaffinity(0))` so it matches the SLURM allocation
(`20260727_run_fig3scan.py:193-205`, `:266`). With 60 tasks on 36 workers the job runs in
two waves, the second set by the slow runs.

**Actual, from `slurm-47863.out`:**

* 60 runs on 36 processes; all 60 completed and all reported solver success.
* **Wall time 1713.1 s ($\approx$ 28.6 min)** against the 72 h limit — comfortably inside.
* Total CPU time $\approx 45{,}244$ s $\approx$ **12.6 core-hours**; mean 754 s per run.
* Per-run cost rises steeply with the amount of coarsening (mean [min–max], seconds):
  $p=0.05$: 114 [43–284]; $p=0.1$: 257 [45–882]; $p=0.25$: 563 [170–998];
  $p=0.5$: 1287 [339–1687]; $p=1$: 1320 [827–1641]; $p=2$: 984 [588–1204].
  Runs near $p \approx 0.5$–$1$ dominate: BDF must resolve every peak-merger event, so
  more coarsening means more implicit steps.
* Memory is modest — each worker holds one $\sim$15 MB solver buffer
  ($3 \times 2048 \times 300$ doubles plus BDF workspace) and discards it before returning;
  only the 300-element peak trace is pickled back. Hence the 150 kB npz.
* The npz is rewritten after every completed run (~1 ms), so a job killed at the wall limit
  still leaves every finished replicate on disk.

---

## 8. Points that are genuinely ambiguous or worth flagging

1. **"Noise" is initial-condition only.** Despite the language of "noise seeds" and
   "replicates", the PDE has no stochastic forcing. Replicates sample initial-condition
   realisations of a deterministic dynamical system.
2. **$l = 1$ makes part of the written model dormant.** All $c_{\rm ext}(1-l)$ terms — in
   the growth rate, the Jacobian, and the steady-state quadratic's constant term — are
   identically zero for the values used. The general equations are implemented, but no
   number reported here probes them.
3. **The "linear instability" reference line is definitional.** As noted in §6.2, it sits
   at 100 because the box was built to hold 100 wavelengths.
4. **The stated bounds of the usable range are comments, not computations.** "Below
   $p \sim 0.05$ the count stays locked" (`run:69-70`), "above $p \sim 2.2$ the instability
   disappears" (same), and "the Turing instability disappears near $K=8$"
   (`functions.py:46-51`) are all assertions in comments; nothing in this folder verifies
   them.
5. **Peak-detection thresholds are hard-coded defaults, never swept.**
   `min_contrast = 0.05` and `prominence_frac = 0.05` (`functions.py:295-296`) are never
   overridden and no sensitivity check exists here. The absolute counts, especially the
   >100 transients and the exact formation times, depend on them.
6. **$t_{\max}$ is not an asymptotic time for $p \ge 1$.** Those traces are still
   descending at $10^8$, so the corresponding points in the `_final` figure are lower
   bounds on coarsening, not steady-state peak counts.
7. **`peaks` is float64 although the counts are integers**, because the array must carry
   NaN for un-run slots. Cast with care.
8. **The first recorded time is $0.1$, not $0$.** `solve_ivp` is given
   `t_span = (t_eval[0], t_eval[-1]) = (0.1, 1e8)`
   (`functions.py:274`), so the initial condition is formally imposed at $t = 0.1$ rather
   than $t = 0$. Immaterial here (the fastest reaction rate is 1), but it means the very
   first row is not literally $u(0)$.
9. **Path mismatch in the sbatch file.** It `cd`s to the cluster path
   `/home/xucapusa/20260709 Spatial Ecology/Fig 3 Minimal Scan` (`:27`), and the
   commented-out local path (`:25`) reads `Fig. 3 Minimal Scan` while this folder is
   `Fig 3 Minimal Scan` (no period). Cosmetic, but it means the sbatch file cannot be run
   from here unchanged.
10. **Replicate index is not comparable across $p$.** Each $(p, \text{rep})$ slot has its
    own seed, so "replicate 3" at $p=0.1$ shares nothing with "replicate 3" at $p=1$. This
    is deliberate but rules out paired comparisons across $p$.
