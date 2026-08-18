# Dispersion relation of the minimal model with fast resources

**Date:** 2026-06-19

We derive a closed-form dispersion relation for the minimal model (1 strain,
2 resources) analysed in `notebooks/`, under the assumptions:

- leakage of the cross-fed (byproduct) resource is zero, `k = 0`;
- the two resources diffuse with the **same** coefficient, `D_I = D_R ≡ D`;
- the resources are **fast** and are adiabatically eliminated ("integrated out").

The strain diffusion coefficient `D_N` is kept general (it turns out to play an
essential role in wavelength selection — see §5).

All Jacobian entries below match the full 3×3 Jacobian assembled in
`ssmc/linstab.py::make_M1` for the parameter map in
`ssmc/minimal_model.py::mmp_to_micrm` (`g = 1`, `w = (1,1)`, `k = 0`). The
result has been checked numerically against the slow eigenvalue of that 3×3
operator in the fast-resource limit.


## 1. Model

Let `N` be the strain density, `I` the primary (influx) resource `R₀`, and `R`
the cross-fed byproduct `R₁`. Specialising the Modified MiCRM reaction term
(`ssmc/rhs.py::_reaction_cell`) to the minimal-model parameters with `k = 0`:

$$
\begin{aligned}
\partial_t N &= D_N\,\nabla^2 N + N\big[(1-l)\,c\,I + d\,R - m\big],\\[2pt]
\partial_t I &= D\,\nabla^2 I + K - r\,I - c\,N\,I,\\[2pt]
\partial_t R &= D\,\nabla^2 R + l\,c\,N\,I - r\,R - d\,N\,R.
\end{aligned}
$$

Symbols (from `MMParams`): `K` primary influx, `m` death rate, `c` primary
consumption, `d` byproduct consumption, `l` primary leakage fraction, `r`
resource dilution. The cross-feeding loop is the term `l c N I`: a fraction `l`
of the primary resource consumed by `N` is excreted as the byproduct `R`, which
`N` then also consumes (gain `d R`).

### Homogeneous steady state

Setting time and space derivatives to zero gives the coexistence state used in
`mm_get_nospace_sol`:

$$
I^* = \frac{K}{r + c\,N^*}, \qquad
R^* = \frac{l\,c\,N^*\,I^*}{r + d\,N^*},
$$

with `N*` the positive root of `c d m\,N² + [m r (c+d) - K c d]\,N + [m r² - K c r (1-l)] = 0`.


## 2. Linearisation

Perturb about the homogeneous state with a single Fourier mode,
`(N,I,R) = (N^*,I^*,R^*) + (\delta N,\delta I,\delta R)\,e^{i q x + \lambda t}`.
Each Laplacian contributes `-q²` times the relevant diffusion coefficient. The
linearised system is `\lambda\,\delta\mathbf{u} = \mathbf{J}(q)\,\delta\mathbf{u}` with

$$
\mathbf{J}(q)=
\begin{pmatrix}
-D_N q^2 & N^*(1-l)c & N^* d\\[2pt]
-c\,I^* & -A(q) & 0\\[2pt]
l c I^* - d R^* & l c N^* & -B(q)
\end{pmatrix},
$$

where we have defined the **resource relaxation rates**

$$
A(q) \equiv r + c\,N^* + D q^2, \qquad
B(q) \equiv r + d\,N^* + D q^2 .
$$

The `(N,N)` entry vanishes because at the coexistence state the strain growth
bracket `(1-l)cI^* + dR^* - m = 0`. Note `D_I = D_R = D` enters `A` and `B`
identically — this is where the equal-diffusion assumption is used.


## 3. Integrating out the fast resources

"Fast resources" means their relaxation rates `A, B` are large compared with the
strain growth rate `|\lambda|`, so the resource perturbations track the strain
field quasi-statically. Setting the resource rows of the linearised system to
zero (i.e. dropping `\lambda\,\delta I`, `\lambda\,\delta R`) and solving:

$$
\delta I = -\frac{c I^*}{A(q)}\,\delta N,
\qquad
\delta R = \frac{1}{B(q)}\Big[(l c I^* - d R^*) + l c N^*\,\delta I/\delta N\Big]\delta N .
$$

Substituting `\delta I` into `\delta R`:

$$
\frac{\delta R}{\delta N}
= \frac{1}{B(q)}\left[\,l c I^* - d R^* - \frac{l c^2 N^* I^*}{A(q)}\,\right].
$$

(Equivalently, this is the Schur complement of the resource block of
`\mathbf{J}(q)`, the standard adiabatic elimination.)


## 4. Closed-form dispersion relation

Insert `\delta I/\delta N` and `\delta R/\delta N` into the strain row
`\lambda\,\delta N = -D_N q^2\,\delta N + N^*(1-l)c\,\delta I + N^* d\,\delta R`:

$$
\boxed{\;
\lambda(q) = -D_N q^2
\;-\;\frac{N^*(1-l)\,c^2 I^*}{A(q)}
\;+\;\frac{N^* d\,(l c I^* - d R^*)}{B(q)}
\;-\;\frac{l\,c^2 d\,(N^*)^2 I^*}{A(q)\,B(q)}
\;}
$$

with `A(q) = r + cN^* + Dq²` and `B(q) = r + dN^* + Dq²`.

The four contributions are physically:

1. `-D_N q²` — direct stabilisation by strain dispersal.
2. `-N^*(1-l)c²I^*/A` — consumption of the primary resource: a local bloom of
   `N` depletes `I`, a **negative** (stabilising) self-feedback, screened over
   the diffusion length of `I`.
3. `+N^* d (l c I^* - d R^*)/B` — the cross-feeding gain minus byproduct
   self-depletion.
4. `-l c² d (N^*)² I^*/(AB)` — the chained primary→byproduct→consumption
   pathway.

Limits:

- `q → 0`: `\lambda(0)` reproduces the slow (resource-eliminated) eigenvalue of
  the non-spatial system; `\lambda(0) < 0` is the condition for the homogeneous
  state to be stable to uniform perturbations.
- `q → ∞`: with `D_N > 0`, `\lambda → -D_N q² → -\infty`. With `D_N = 0`,
  `\lambda → (Q-P)/(Dq²) → 0`, approached from **above** when `Q > P` and from
  below when `Q < P` (constants `P, Q` defined in §5). This sign of approach is
  the crux of the `D_N = 0` behaviour discussed below.


## 5. Instability with finite (or small) `D_N`

It is convenient to write the dispersion relation as a single rational function.
Put `x ≡ Dq² ≥ 0`, `ρ ≡ D_N/D` (so `D_N q² = ρx`), `a₀ ≡ r + cN^*`,
`b₀ ≡ r + dN^*`, and

$$
P = N^*(1-l)c^2 I^* > 0,\quad
Q = N^* d\,(l c I^* - d R^*),\quad
S = l c^2 d (N^*)^2 I^* > 0 .
$$

Then `A = a₀ + x`, `B = b₀ + x`, and multiplying by `AB > 0` (which preserves the
sign of `λ`, hence the unstable band `{λ>0}`):

$$
A B\,\lambda(x) \;=\; \underbrace{-\rho\,x\,(a_0+x)(b_0+x)}_{\text{strain diffusion}}
\;+\;(Q-P)\,x \;+\; C_0,
\qquad C_0 \equiv Q a_0 - P b_0 - S .
$$

The intercept is `C_0 = A(0)B(0)\,λ(0)`, negative when the homogeneous state is
stable to uniform perturbations.

### 5.1 The `D_N = 0` case is *not* monotonic

Setting `ρ = 0` gives `ABλ = (Q-P)x + C_0`, a straight line, so

$$
\lambda(x)\big|_{D_N=0} = \frac{(Q-P)\,x + C_0}{(a_0+x)(b_0+x)} .
$$

This is **not** monotonically increasing. Its derivative has the sign of a
*downward* parabola,

$$
\lambda'(x) \;\propto\; -(Q-P)\,x^2 \;-\; 2C_0\,x \;+\;\big[(Q-P)\,a_0 b_0 - C_0(a_0+b_0)\big],
$$

so on `x ≥ 0` there is **at most one interior maximum**. Concretely, when
`Q > P` and `C_0 < 0` the curve:

1. starts negative at `λ(0) = C_0/(a₀b₀) < 0`;
2. crosses zero at `x_c = -C_0/(Q-P)` (i.e. `q_c = √(x_c/D)`);
3. rises to a single finite peak at
   `x_★⁰ = [ -C_0 + √(C_0² + (Q-P)((Q-P)a₀b₀ - C_0(a₀+b₀))) ] / (Q-P)`;
4. then **decays back toward `0⁺`** as `x → ∞`, since `λ ≈ (Q-P)/x`.

So even at `D_N = 0` there is a finite fastest-growing wavenumber `q_★ = √(x_★⁰/D)`
— the earlier claim of a monotonic "ultraviolet" growth was wrong. The genuine
pathology at `D_N = 0` is different: because `λ → 0⁺`, the **unstable band is
unbounded**, `λ(q) > 0` for *all* `q > q_c`. Every short wavelength is (weakly)
unstable.

### 5.2 Turing threshold

A finite-wavelength instability needs `λ(0) < 0` and `max_{q>0} λ(q) > 0`. At
`ρ = 0` the second condition reduces (given `C_0 < 0`) simply to a positive
slope `Q > P`. Writing this out with `R^* = l c N^* I^*/b₀` collapses to

$$
\boxed{\;\dfrac{d\,l\,r}{\,r + d\,N^*\,} \;>\; (1-l)\,c \;}\qquad(D_N \to 0)
$$

— the cross-feeding return (consumption `d`, leakage `l`, weighted by the
fraction `r/(r+dN^*)` of byproduct surviving dilution) must beat the net
primary-resource uptake `(1-l)c`.

### 5.2.1 Critical abundance — fully closed form

`N^*` is itself closed-form: the coexistence branch is the larger root

$$
N^* = \frac{-q_b + \sqrt{q_b^2 - 4 q_a q_c}}{2 q_a},\quad
q_a = cdm,\;\;
q_b = mr(c+d) - Kcd,\;\;
q_c = mr^2 - Kcr(1-l).
$$

Because the threshold's left side `dlr/(r+dN^*)` is **monotonically decreasing in
`N^*`**, the inequality can be inverted into a *critical abundance*. Setting
`dlr/(r+dN^*) = (1-l)c` and solving for `N^*`,

$$
N^*_{\rm crit} \;=\; \frac{l r}{(1-l)c} - \frac{r}{d}
\;=\; \frac{r\,[\,d l - (1-l)c\,]}{(1-l)\,c\,d},
$$

so the entire Turing condition (at `D_N → 0`) is just a comparison of two
closed-form abundances:

$$
\boxed{\;\textbf{instability}\iff N^* < N^*_{\rm crit}\;}
$$

Substituting the root for `N^*` gives the boundary explicitly in the bare
parameters. The split is interpretable: `N^*(K,m,c,l,d,r)` is *how abundant the
consumer actually is*, while `N^*_{\rm crit}(l,c,d,r)` is *how abundant it may be
before the cross-feeding field can pattern*. (Requires `dl > (1-l)c` for
`N^*_{\rm crit} > 0`, else no instability at any abundance.)

**`K` and `m` enter only through `N^*`.** `N^*_{\rm crit}` contains neither.
Raising `K` raises `N^*` (for large `K`, `N^* ∼ K/m`) toward the fixed ceiling
`N^*_{\rm crit}`, so influx **suppresses** the instability: a denser population
homogenises the cross-feeding field rather than patterning it. Near the
extinction boundary `N^* → 0` the condition is always met (provided
`dl > (1-l)c`).

**Why the threshold barely depends on `d`.** Both sides of `N^* < N^*_{\rm crit}`
carry the *same* `-r/d` saturation and approach constants as `d→∞`:

$$
N^* \;\to\; \frac{K}{m}-\frac{r}{c},
\qquad
N^*_{\rm crit} \;=\; \frac{lr}{(1-l)c}-\frac{r}{d}\;\to\;\frac{lr}{(1-l)c}.
$$

So once consumption is fast enough that `dN^* ≫ r`, the gap between `N^*` and
`N^*_{\rm crit}` stops moving with `d` and the criterion saturates to a
`d`-independent relation. Substituting the limits into `N^* < N^*_{\rm crit}`
collapses remarkably cleanly to

$$
\boxed{\; m r \;>\; (1-l)\,K c \;}
\qquad\Longleftrightarrow\qquad
K < \frac{mr}{(1-l)c}
\qquad (dN^* \gg r).
$$

Mechanistically: the destabilising feedback is the strain's byproduct-derived
growth `dR^* = l c I^* \varphi`, with `\varphi \equiv dN^*/(r+dN^*) \in (0,1)` the
fraction of leaked byproduct *recaptured* rather than washed out by dilution.
`d` enters only through `\varphi`, which saturates at 1 — speeding up consumption
beyond full recapture cannot enlarge the cross-feeding gain `dR^* → lcI^*`, so it
cannot change whether patterns form. `d` matters only in the marginal window
`dN^* ∼ r`; any parameter set admitting a coexistence state at all sits in the
saturated regime, which is why the criterion looks essentially `d`-independent in
practice. (Note this is about *whether* patterns form; the selected *wavelength*
still depends on `d` through `b₀ = r + dN^*` in `B(q)`.)

### 5.3 Role of finite `D_N`: UV cutoff and threshold shift

Restoring `ρ > 0` makes `ABλ` a **cubic with negative leading coefficient**
`-ρx³`, so now `λ → -∞` as `q → ∞`. Two consequences:

- **Bounded unstable band.** The band `{λ>0}` becomes a finite interval
  `[q_-, q_+]`. The upper (ultraviolet) edge comes from balancing `-ρx³` against
  `(Q-P)x` at large `x`:

  $$
  x_+ \approx \sqrt{\frac{Q-P}{\rho}}
  \quad\Longrightarrow\quad
  q_+ \approx \left(\frac{Q-P}{D\,D_N}\right)^{1/4}.
  $$

  This diverges as `D_N → 0`, recovering the unbounded band of §5.1. *This* is
  what finite `D_N` supplies: a short-wavelength cutoff, not the selected
  wavelength.

- **The fastest-growing mode is `D_N`-insensitive for small `D_N`.** Because the
  peak `x_★⁰` of §5.1 is already finite, adding `-ρx` perturbs it only weakly.
  By the envelope theorem the peak *value* shifts linearly,

  $$
  \lambda_{\max}(D_N) \approx \lambda_{\max}^{(0)} - \frac{D_N}{D}\,x_\star^0 ,
  $$

  with the peak *location* `q_★` essentially fixed at its `D_N = 0` value until
  `D_N` grows large enough that `q_+` collapses onto `q_★`. The instability
  therefore survives provided

  $$
  D_N \;\lesssim\; D\,\frac{\lambda_{\max}^{(0)}}{x_\star^0},
  $$

  a clean small-`D_N` correction to the §5.2 threshold.

In short: the pattern wavelength is set by the **resource sector**
(`a₀, b₀, P, Q, S`) and is robust to small strain diffusion; `D_N` controls only
the ultraviolet tail and, through the bound above, how much margin beyond
`Q > P` is needed. The mechanism works despite `D_I = D_R`: the relevant
diffusion separation is between the fast, diffusing resources and the slow,
weakly-diffusing consumer.


## 6. Numerical check

**Closed form vs. full operator.** For `K=12, m=1, c=1, l=0.7, d=1.3, r=1`,
`D=1`, the boxed `λ(q)` reproduces the largest-real-part eigenvalue of the full
3×3 operator `M1 - q² diag(D_N, D, D)` (`ssmc/linstab.py`) in the fast-resource
limit: the two curves converge as the resource timescale → 0 at every `q`.

**Finite-`D_N` predictions.** For a genuinely Turing-unstable set
`K=3.503, m=1, c=0.600, l=0.687, d=1.684, r=1` (found via `scan_k`), the
reduced constants are `P=0.220, Q=0.562, S=0.683`, `C_0=-0.369`, so `Q>P`
(unstable) and `λ(0)=-0.102<0`. The `D_N=0` curve is non-monotonic exactly as
§5.1 predicts: zero crossing `q_c=1.04`, finite peak `q_★=2.02` with
`λ_max⁰=0.0283`, then decay to `0⁺` (unbounded band). Adding `D_N`:

| `D_N/D` | `q_+` (numeric) | `((Q-P)/(D D_N))^{1/4}` | `q_★` | `λ_max` (numeric) | `λ_max⁰ - (D_N/D)x_★⁰` |
|--------:|----------------:|------------------------:|------:|------------------:|-----------------------:|
| `10⁻⁵`  | 13.5            | 13.6                    | 2.018 | 0.02829           | 0.02829                |
| `10⁻⁴`  | 7.48            | 7.65                    | 2.004 | 0.02793           | 0.02792                |
| `10⁻³`  | 3.99            | 4.30                    | 1.893 | 0.02452           | 0.02425                |

(`q_+ = √(x_+/D)`.) The UV band edge follows the `(D D_N)^{-1/4}` law, the peak
location is nearly frozen, and the peak value tracks the linear envelope
estimate — all as derived in §5.3.


## 7. Unequal resource diffusion (`D_I ≠ D_R`)

Everything in §§1–4 goes through unchanged; the equal-diffusion assumption was
used **only** to write `A` and `B` with a common `Dq²`. Relaxing it, the two
resources simply carry their own coefficients:

$$
A(q) = r + cN^* + D_I\,q^2, \qquad
B(q) = r + dN^* + D_R\,q^2,
$$

and the boxed closed-form dispersion relation of §4 is *identical* with these
`A`, `B`. (Verified: the closed form still reproduces the slow eigenvalue of the
full 3×3 operator `M1 - q²\,diag(D_N, D_I, D_R)` in the fast-resource limit.)
What changes is the stability analysis, because `A` and `B` no longer share a
single variable `x = Dq²`.

### 7.1 Threshold acquires the diffusion ratio

Write `y = q² ≥ 0` and multiply `λ` by `AB > 0`. The reaction-plus-resource-
diffusion part (i.e. `D_N = 0`) is again **linear in `y`**, but the slope now
mixes the two coefficients:

$$
A B\,\lambda\big|_{D_N=0} = \big(Q\,D_I - P\,D_R\big)\,y + C_0 ,
\qquad C_0 = Q a_0 - P b_0 - S
$$

(the intercept `C_0` is diffusion-independent, since diffusion vanishes at
`q = 0`). With `C_0 < 0` (homogeneous stability), instability requires a positive
slope, `Q D_I > P D_R`, i.e.

$$
\frac{D_R}{D_I} < \frac{Q}{P} = \frac{d\,l\,r}{(1-l)\,c\,(r + dN^*)} .
$$

Equivalently the §5.2 threshold just picks up the diffusion ratio on its
right-hand side:

$$
\boxed{\;\dfrac{d\,l\,r}{\,r + d\,N^*\,} \;>\; (1-l)\,c\;\dfrac{D_R}{D_I}\;}
$$

Setting `D_I = D_R` recovers §5.2 exactly. The critical abundance generalises to

$$
N^*_{\rm crit} = \frac{l r}{(1-l)c}\,\frac{D_I}{D_R} - \frac{r}{d},
\qquad
\textbf{instability}\iff N^* < N^*_{\rm crit},
$$

and the saturated (`dN^* ≫ r`) form becomes

$$
K \;<\; \frac{mr}{c}\left(1 + \frac{l}{1-l}\,\frac{D_I}{D_R}\right)
\qquad\xrightarrow{\;D_I=D_R\;}\qquad K < \frac{mr}{(1-l)c}.
$$

### 7.2 Interpretation: the classic activator–inhibitor ratio

The byproduct `R` is the **activator**: a local bloom of `N` excretes more `R`,
which `N` consumes (`+dR` in its growth), a positive feedback. The primary
resource `I` is the **inhibitor**-like species: a bloom of `N` *depletes* `I`
locally (`-cNI`), suppressing growth. The standard Turing rule — *inhibitor must
diffuse faster than activator* — is exactly the condition above:

- `D_R < D_I` (byproduct/activator slower than primary/inhibitor) **lowers** the
  right-hand side and **promotes** patterning;
- `D_R > D_I` **raises** the bar and suppresses it.

With `D_I = D_R` the resource–resource ratio is neutral and the instability is
carried entirely by the *third* slow species, the weakly-diffusing consumer `N`
(§5). Unequal resource diffusion adds the familiar two-species activator–
inhibitor channel on top of that.

### 7.2.1 Reading `N^*_crit` with unequal diffusion

$$
N^*_{\rm crit} = \frac{lr}{(1-l)c}\,\frac{D_I}{D_R} - \frac{r}{d}.
$$

The criterion is still an **abundance ceiling** (`instability ⇔ N^* < N^*_crit`),
but the diffusion ratio rescales it multiplicatively. The clean conceptual split:

- `N^*` is the *well-mixed* steady state and is **diffusion-independent**;
- `N^*_crit` is a property of the *spatial* threshold and scales as `D_I/D_R`.

So tuning `D_I/D_R` does not touch the realised abundance — it slides the ceiling
up or down past a fixed `N^*`. Raising `D_I/D_R` lifts the ceiling and brings
patterning within reach of denser populations (larger `K`); lowering it presses
the ceiling below `N^*` and switches patterning off.

The leading factor `\frac{l}{1-l}\cdot\frac{r}{c}` is the equal-diffusion ceiling
(cross-feeding investment ratio `l/(1-l)` times the resource turnover/uptake
scale `r/c`); `D_I/D_R` multiplies it; and `-r/d` is the incomplete-recapture
correction that vanishes as `d→∞`. The limiting cases fix the activator/inhibitor
identification:

| limit | `N^*_crit` | consequence |
|---|---|---|
| `D_R → 0`   (byproduct immobile)        | `→ +∞`     | always unstable — a non-diffusing **activator** is the ideal local self-reinforcer |
| `D_R → ∞`   (byproduct instantly shared) | `→ -r/d<0` | never unstable — a globally shared activator gives no *local* feedback |
| `D_I → ∞`   (primary instantly replenished) | `→ +∞`  | always unstable — inhibitory depletion is smoothed out, cannot stabilise short λ |
| `D_I → 0`   (primary depletion stays local) | `→ -r/d<0` | never unstable — strong local self-limitation kills patterns |

i.e. the byproduct `R` (activator) wants to be **slow/local** and the primary `I`
(inhibitor) wants to be **fast/long-range** — "local activation, long-range
inhibition". Equivalently, `D_I/D_R` renormalises the effective cross-feeding
strength: the §5.2 contest of cross-feeding return `dlr/(r+dN^*)` against primary
uptake `(1-l)c` becomes a contest against the *discounted* uptake
`(1-l)c\,(D_R/D_I)`.

### 7.3 What does *not* change

`D_N` is still required for an ultraviolet cutoff: at `D_N = 0` the numerator
`(Q D_I - P D_R)y + C_0` is linear while the denominator `AB ∼ D_I D_R y²`, so
`λ → 0` as `q → ∞` and the unstable band is unbounded exactly as in §5.1. The
finite-`D_N` band edge of §5.3 generalises to
`q_+ ≈ ((Q D_I - P D_R)/(D_I D_R\,D_N))^{1/4}`. So `D_R/D_I` sets *whether* (and
how strongly) the system patterns; `D_N` still sets the short-wavelength cutoff.

### 7.4 Numerical check

For the §6 unstable set (`Q/P = 2.55`) with `D_I = 1`, `D_N = 10⁻⁴`, a `scan_k`
sweep over `D_R` flips from `unstable` to `stable` as `D_R/D_I` crosses `≈ 2.5`,
matching the predicted `D_R/D_I < Q/P = 2.55` (the onset sits just below 2.55
because finite `D_N` demands a little extra margin, per §5.3). The closed-form
`λ(q)` with `D_I ≠ D_R` agrees with the full 3×3 slow eigenvalue to 4–5 digits in
the fast-resource limit.


## 8. Predicting the final (nonlinear) patterned state

Everything in §§4–7 is **linear**: it predicts *whether* the uniform state loses
stability, the onset wavenumber, and growth rates — but says nothing about the
amplitude, shape, or biomass of the state the system actually settles into. In
particular, `N^*_crit` is a threshold on the *background* abundance, **not** a
saturation amplitude: the nonlinear peak biomass is set by local resource supply
and is generally well above `N^*_crit`. Predicting the final state is a separate,
nonlinear problem. The tools, from exact to approximate:

### 8.1 The exact object: a reduced nonlocal equation

Eliminate the fast resources *nonlinearly* (not just their linearisation). For a
frozen consumer field `N(x)`, the quasi-static resources obey **linear
screened-Poisson (Yukawa) problems**

$$
D_I\nabla^2 I - (r + cN(x))\,I = -K,
\qquad
D_R\nabla^2 R - (r + dN(x))\,R = -\,l c\,N(x)\,I .
$$

Their solutions `I[N](x)`, `R[N](x)` are convolutions of the sources with a
Green's function whose **local screening length is `√(D_I/(r+cN))`** — short
where `N` is high. The patterned steady states are then the nonconstant solutions
of the single nonlocal equation

$$
0 = D_N\nabla^2 N + N\big[(1-l)c\,I[N] + d\,R[N] - m\big].
$$

Every method below is a strategy for solving this.

### 8.2 Near onset: weakly nonlinear theory (carried out)

Work with the full 3-field deviation `u = (n, i, ϱ)` from `(N^*, I^*, R^*)`. The
system is

$$
\partial_t u = L(\nabla)\,u + Q(u,u),
$$

where `L` is the linear operator whose Fourier symbol is the §3/§7 Jacobian
`J(q)` (the matrix with `−D_N q²`, `A(q)`, `B(q)`), and — crucially — the
kinetics are **purely quadratic** (no cubic terms). The symmetric bilinear is
local:

$$
Q(a,b) =
\begin{pmatrix}
(1-l)c\,\overline{a_N b_I} + d\,\overline{a_N b_R}\\[2pt]
-c\,\overline{a_N b_I}\\[2pt]
l c\,\overline{a_N b_I} - d\,\overline{a_N b_R}
\end{pmatrix},
\qquad \overline{a_N b_X} \equiv \tfrac12(a_N b_X + a_X b_N).
$$

**Onset.** Tune a control parameter (take `K`) to the value `K_c` where
`\max_q \max\operatorname{Re}\operatorname{eig} J(q) = 0`, with marginal
wavenumber `q_c`. Let `e` be the right null vector of `J(q_c)` and `e^†` the left
null vector, normalised `e^† \cdot e = 1`.

**Multiple scales.** With `u = ε u_1 + ε^2 u_2 + …`, slow time `T = ε^2 t`, and
`μ ≡ K_c − K = ε^2 μ_2`, the leading mode is
`u_1 = A(T)\,e\,e^{i q_c x} + \text{c.c.}` The quadratic term forces a mean (`q=0`)
and a harmonic (`2q_c`) response,

$$
u_2 = |A|^2\,M + \big(A^2 H\,e^{2 i q_c x} + \text{c.c.}\big),\quad
M = -2\,J(0)^{-1} Q(e,e),\;\; H = -\,J(2q_c)^{-1} Q(e,e)
$$

(`Q(e,\bar e) = Q(e,e)` because `e` is real, and `J(0)`, `J(2q_c)` are invertible
since only `q_c` is marginal). The `O(ε^3)` solvability condition (projection onto
`e^†`) gives a **Stuart–Landau equation**

$$
\frac{dA}{dT} = \sigma A - g\,|A|^2 A,\qquad
\boxed{\,g = 4\,e^{†}\!\cdot Q\big(e,\,J(0)^{-1}Q(e,e)\big)
        + 2\,e^{†}\!\cdot Q\big(e,\,J(2q_c)^{-1}Q(e,e)\big)\,}
$$

with `σ = (∂λ/∂μ)|_{q_c} > 0`. If `g > 0` the bifurcation is **supercritical**:
saturated stripes with `|A| = √(σμ/g)` and peak consumer deviation
`≈ 2|e_N|\,|A|`. The restoring effect is the Monod-like fall of `I[N]≈K/(r+cN)` —
a clump starves itself of primary resource — partly offset by the cross-feed
`R[N]`; `g` is the net curvature of that competition.

**Evaluated for the §6 set** (`m=1, c=0.6, l=0.687, d=1.684, r=1`,
`D_N=10⁻³, D_I=D_R=1`): onset at `K_c ≈ 3.786`, `q_c ≈ 2.863`, marginal vector
`e ≈ (-0.992,\,0.127,\,-0.018)` (consumer up ⇒ primary down, as expected), and

$$
g \approx -0.012 \;<\; 0 \quad\Rightarrow\quad \textbf{subcritical}.
$$

So the cubic does *not* saturate the pattern: amplitude jumps to finite size with
hysteresis, and the small-amplitude formula `√(σμ/g)` does not apply (it would be
imaginary). Saturation is set at quintic order / by the strongly nonlinear spike
balance of §8.3.

**2D — hexagons.** With a quadratic nonlinearity and no symmetry, a resonant
triad of modes (three `|q|=q_c` at `120°`, summing to zero) couples at quadratic
order. The hexagon amplitude equations carry a coefficient
`γ = 2\,e^{†}\!\cdot Q(e,e)`, here `≈ 0.014 ≠ 0`. A nonzero `γ` makes **hexagons /
spots bifurcate transcritically** — they exist for `μ` slightly *below* linear
onset and generically win over stripes near threshold. Both the explicit `g<0`
and the `γ≠0` point to the same picture: **subcritical onset, hexagonal/spotted
patterns, hysteresis.**

### 8.3 Far from onset: spikes (carried out)

Because `D_N ≈ 0` and the bifurcation is subcritical (§8.2), the relevant states
are strongly nonlinear **spikes**, not small sinusoids. Set up a single 1D spike
of the consumer at `x=0` in the fast-resource reduction.

**Outer resource fields (Yukawa).** Away from and across the spike the resources
solve the screened-Poisson problems of §8.1, linear in `I`, `R` for given `N(x)`.
The 1D Green's function of `(a_0 - D_I\partial_x^2)` is

$$
G_I(x) = \frac{1}{2\sqrt{D_I a_0}}\,e^{-|x|/\ell_I},
\qquad \ell_I = \sqrt{D_I/a_0},\quad a_0 = r + c\langle N\rangle,
$$

and likewise `G_R` with `ℓ_R = √(D_R/b_0)`, `b_0 = r + d⟨N⟩`. So `ℓ_I`, `ℓ_R` are
the **resource screening lengths** — the range over which a spike perturbs each
resource. The spike depletes `I` (a sink `∝ cNI`) and sources `R` (`∝ lcNI`)
within `±ℓ_{I,R}`.

**Inner spike width.** Inside the spike the consumer balances its own diffusion
against the *net* local kinetics. In the depleted core the growth bracket is
dominated by `−m` (resources drawn down), so `D_N N'' ≈ mN` sets

$$
w \sim \sqrt{D_N/m}\ \xrightarrow{\,D_N\to0\,}\ 0 .
$$

The spike is sharp, with width vanishing as `√D_N`.

**Spike height from the biomass budget.** Integrating the steady PDEs over one
pattern cell (length `L`, periodic) kills every diffusion and gradient term and
leaves an **exact identity** (see §8.4):

$$
m\,\langle N\rangle = K - r\,(\langle I\rangle + \langle R\rangle).
$$

So the *mean* biomass is fixed by the resource budget, not by the pattern. A
spike concentrates the cell's biomass `M = \langle N\rangle L` into width `w`, so
its height obeys `N_{\rm pk}\,w \sim M`, giving

$$
\boxed{\;\frac{N_{\rm pk}}{\langle N\rangle} \sim \frac{L}{w}
\sim \ell_I\sqrt{\frac{m}{D_N}}
= \sqrt{\frac{m\,D_I}{a_0\,D_N}}\;}
$$

where the spacing `L` is set by the resource screening length `ℓ_I` (spikes repel
over the range they deplete the shared primary resource). The key qualitative
prediction: as the consumer diffusion `D_N → 0`, **spikes grow taller and
narrower without bound**, `N_{\rm pk} ∝ D_N^{-1/2}`, while the mean stays pinned
by the budget. Peak `N ≫ N^*_crit`, confirming that the linear threshold is not a
saturation amplitude.

Exact prefactors (and the precise spacing-selection / spike-interaction laws)
require matched asymptotics in the Gierer–Meinhardt "semi-strong" style; the
boxed relations are the controlling scalings.

### 8.4 Exact global constraints

Integrating the steady PDEs over the domain (periodic / no-flux) kills all
diffusion terms. Combining the three integrated equations — adding `(1-l)×` and
`l×` of the consumption flux so the cross-feed cancels — yields the **exact
biomass budget**

$$
\boxed{\; m\,\langle N\rangle = K - r\big(\langle I\rangle + \langle R\rangle\big) \;}
$$

independent of the pattern's shape: mean biomass = influx minus what dilution
washes away, per unit death rate. (Checked against the homogeneous state for the
§8.2 onset parameters: `m N^* = 1.307` vs `K - r(I^*+R^*) = 1.307`.) When
resources are efficiently consumed (`⟨I⟩,⟨R⟩` small), `⟨N⟩ → K/m` — maximal
productivity. This pins `⟨N⟩` and, with §8.3, bounds the peaks, even though it
gives no profile.

### 8.5 Numerical continuation and multistability

For the actual branch structure, **pseudo-arclength continuation** (AUTO,
`pde2path`) on the reduced equation traces patterned steady states vs `K` or
`D_R/D_I`, locating folds, subcritical regions, and **localised states /
homoclinic snaking** (isolated patches coexisting with the uniform state). If the
bifurcation is subcritical the final state is **history-dependent** — there is no
closed-form answer and initial conditions matter.

### 8.6 Wavelength caveat

The final spacing is *not* simply `2π/q_★`. After the linear transient,
**Eckhaus / phase instabilities and coarsening** readjust the wavelength, usually
toward something longer than the linearly fastest mode.

### 8.7 Summary of the two predictions

- **(a) Near onset (§8.2):** the bifurcation for the §6 parameters is
  **subcritical** (`g ≈ -0.012 < 0`) with a nonzero hexagon coupling
  (`γ ≈ 0.014`) — expect **spots/hexagons appearing with hysteresis**, not
  small-amplitude stripes. The Stuart–Landau coefficient `g` is given in closed
  form (projection formula) and is cheap to re-evaluate for any parameter set.
- **(b) Far from onset (§8.3):** the states are **sharp spikes** of width
  `w ∼ √(D_N/m)` spaced by the resource screening length `ℓ_I = √(D_I/a_0)`, with
  peak `N_{\rm pk}/⟨N⟩ ∼ √(m D_I/(a_0 D_N))` growing as `D_N^{-1/2}`, while the
  mean biomass is pinned by the exact budget `m⟨N⟩ = K - r(⟨I⟩+⟨R⟩)`.

Because `D_N ≈ 0` and the onset is subcritical, the 2D simulations most likely
sit in the **(b) spike** regime (steep spots) rather than the weakly-nonlinear
regime — consistent with subcritical jumps straight to finite-amplitude
structures.

### 8.8 Validation against the 2D simulation

Tested against `20260618_09_sol_t4000.npz` (final field, `t=4000`). That run is a
*random community* (3 surviving strains + 30 resources), **not** the minimal
model, so it probes the *general* spike-regime predictions rather than the
model-specific coefficients — but it sits squarely in the relevant regime:
`D_N = 10⁻⁶`, byproduct diffusion `0.01`, influx diffusion `1.0` (note
`D_byp < D_influx` — the pattern-promoting direction of §7). The total biomass
field (`128²`, `dx=0.023`) gives:

| prediction | theory | simulation |
|---|---|---|
| pattern morphology (§8.2: subcritical, hexagons/spots) | spots | **~47 spots** (area frac 0.11) |
| spacing (§8.6) `≈` linear Turing `L_★` | `L_★ = 0.493` | dominant `λ ≈ 0.486` (ratio 0.99) |
| screening-length structure (§8.1) | fast resource smooth, slow resource spiky & co-located with biomass | influx (`D=1`) smooth; byproducts (`D=0.01`) spiky on the spikes ✓ |
| concentration `N_pk/⟨N⟩` (§8.3, **2D spots** `∼ (L_★/w)²`) | `(0.493/0.161)² ≈ 9.4` | **peak/mean = 9.5** |

The 2D peak/mean lands almost exactly on the spot version of the §8.3 scaling
(in 2D the area fraction is `(w/L)²`, so `N_pk/⟨N⟩ ∼ (L/w)²`, replacing the 1D
`L/w`). One refinement the data forces: the spike width here is `w ≈ 0.16`, set by
the **slow-byproduct screening length** `ℓ_byp = √(D_byp/b_0) ∼ 0.07`–`0.16`,
**not** by `√(D_N/m) ∼ 10⁻³` (which is sub-grid). So the §8.3 trend "spikes
sharpen as `D_N→0`" is correct but **floors at the slowest resource screening
length**: once `√(D_N/m)` drops below `ℓ_R`, the consumer can be no sharper than
the resource field that drives it, and `w → ℓ_R`. The effective width to use in
the concentration estimate is therefore `w ∼ max(√(D_N/m),\,ℓ_R)`.
