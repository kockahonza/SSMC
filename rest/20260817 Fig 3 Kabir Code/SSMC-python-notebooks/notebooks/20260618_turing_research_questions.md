# Turing Patterns in Microbial Communities — Research Questions

A working catalogue of research questions for a paper on Turing pattern formation in a
spatial Modified MiCRM (microbial consumer–resource model). Each entry gives a short
rationale and a concrete note on how the current `ssmc` codebase addresses it, or what
would need to be built. Questions are grouped by theme; a recommended first-paper
storyline is given at the end.

## Codebase capability map (what exists today)

- **Reaction model** (`ssmc/rhs.py`, `_reaction_cell`): Modified MiCRM with consumption
  `c`, leakage `l`, byproduct routing tensor `D` (`b -> a`), death `m`, influx `K`,
  dilution `r`, energy density `w`, efficiency `g`.
- **Linear stability** (`ssmc/linstab.py`): exact Jacobian `make_M1`, dispersion
  `M(k) = M1 - k² diag(Ds)`, `scan_k` → dispersion array + classification code,
  `classify` → `stable` / `unstable` (Turing) / `nospace_unstable`. This is the analytic
  workhorse and is **cheap** (dense eigensolve on an `(Ns+Nr)` matrix per `k`).
- **Sampling** (`ssmc/sampling.py`, `JansSampler3`): random communities with per-field
  distributions (Dirac / Gaussian / scipy frozen), single- or multi-influx, tunable
  `prob_eating`, `num_byproducts`, leakage, diffusion (`Ds`, `Dr`, `Drinflux`).
- **Solvers** (`ssmc/solver.py`): `solve_steady_state` (homogeneous FP via BDF),
  `solve_pde` (1D, BDF + analytic Jacobian sparsity), `solve_pde_2d` (BDF),
  `solve_pde_2d_imex` (SBDF2 IMEX-FFT, periodic only — the scalable long/large-run path).
- **Reduction** (`ssmc/reduce.py`): exact pruning of extinct strains (`N_i=0` invariant).
- **Minimal model** (`ssmc/minimal_model.py`) + closed-form analytics
  (`ssmc/analytics.py`): 1-strain / 2-resource (primary + byproduct), with analytic
  extinction/instability boundaries and `k*²` formulas in `(l, p=D_R/D_I, β)`.

**Gaps the questions repeatedly hit** (flagged as "needs new capability"):
parameter continuation/branch tracking; automated ensemble sweep harness over seeds;
amplitude-equation / weakly-nonlinear reduction; immigration/source terms in the RHS;
spatially heterogeneous parameters; quantitative pattern-morphology metrics; noise terms
(stochastic PDE). None of these is large, but none exists yet.

---

## Theme A — Mechanism: what makes a community Turing-unstable

**A1. Which diffusion ratio drives the instability, and is there a sharp threshold?**
The mechanism is differential diffusion: fast influx resource (`Drinflux≈1`), slower
byproducts (`Dr`), near-immobile strains (`Ds≈1e-6`). *Rationale:* classical Turing
needs an activator/inhibitor diffusion gap; here the "inhibitor" is the diffusing
resource and the "activator" is the sessile strain. *Code:* directly answerable now —
sweep `Ds`, `Dr`, `Drinflux` in `SpatialMiCRMParams` and call `scan_k`; the minimal-model
`fr2_*` analytics in `analytics.py` give the closed-form instability band in `p = D_R/D_I`
to validate against. No simulation needed.

**A2. Is leakage `l` necessary, and is there an optimal leakage for instability?**
Byproducts (hence cross-feeding) only exist when `l>0`; the analytics
(`fr2_beta_lb`, `fr2_beta_ub`) show the instability band in `l` is bounded above and
below. *Rationale:* establishes cross-feeding as the engine, not a side effect. *Code:*
`scan_k` over an `l` grid (full model) plus `fr_cor1_instab_line_K` / `fr2_instab_beta_range`
(minimal model) — both exist. Strong analytic-vs-numeric validation opportunity.

**A3. How does byproduct-network topology gate pattern formation?**
`D` routes byproduct `b->a`; `num_byproducts` and `prob_eating` set network density.
*Rationale:* tests whether Turing instability is a property of specific motifs (linear
cascade vs branched vs cyclic cross-feeding) rather than of bulk parameters. *Code:*
`JansSampler3` controls `num_byproducts`/`prob_eating`; classify ensembles with `scan_k`.
*Needs:* a small motif-construction helper and a network-statistic extractor to correlate
graph features (cycles, path length, modularity) with instability probability — modest
addition on top of existing sampling.

**A4. Single-influx vs multi-influx: does forcing the byproduct cascade matter?**
Single-influx makes all but one resource purely metabolic, so the spatial structure must
propagate down the cross-feeding chain. *Rationale:* this is the biologically clean
"one carbon source, emergent niches" setup and the basis of notebooks 03/04/09. *Code:*
`num_influx_resources` in the sampler; compare instability rate / `k*` distributions
between `num_influx=1` and larger. Directly answerable now.

**A5. Can the full-model instability be predicted from a 3-species reduced model?**
After pruning, only ~3 strains survive (empirically). *Rationale:* a reduced
3-strain / few-resource model would make the mechanism analytically tractable and connect
to the existing minimal-model analytics. *Code:* `reduce.prune_strains` gives the exact
survivor subsystem; minimal model handles 1×2. *Needs:* extend `analytics.py` toward a
2–3 strain semi-symbolic treatment (new derivation), or at least numerically reduce and
re-run `scan_k` on the survivor system to confirm the dispersion relation is preserved.

---

## Theme B — Community assembly × pattern formation

**B1. Does spatial pattern formation change which strains coexist vs the well-mixed case?**
*Rationale:* the headline ecological claim — patterns could rescue strains that go
extinct at the homogeneous fixed point, or kill ones that survive it. *Code:* compare
`survivor_mask` at the homogeneous FP against the set of strains with non-trivial biomass
in a long PDE run. The pruning docstring explicitly flags "pattern-induced local
invasion" as a caveat to check. *Needs:* the IMEX 2D solver for long runs, plus a
re-invasion test (seed a dropped strain at low density into the patterned state and check
its growth) — small driver code.

**B2. Does spatial structure rescue diversity (more coexisting strains than well-mixed)?**
*Rationale:* spatial niching is a classic diversity-maintenance mechanism; quantify it
here. *Code:* ensemble of sampled communities, count survivors well-mixed
(`solve_steady_state`) vs patterned (long PDE). *Needs:* ensemble harness + the
re-invasion test from B1; long 2D runs via `solve_pde_2d_imex`.

**B3. How does Turing-instability probability scale with community richness `Ns`, `Nr`?**
*Rationale:* connects to random-matrix / large-ecosystem theory (does instability become
generic or vanishing as systems grow?). *Code:* `JansSampler3` + `scan_k` over a grid of
`(Ns, Nr)`; cheap because it is linear-stability only. *Needs:* ensemble sweep harness
(loop + bookkeeping); this is the single most reusable missing piece.

**B4. What community-level features predict instability (a phenomenological classifier)?**
*Rationale:* if we can predict `unstable` from cheap summary statistics (number of
survivors, total leakage, mean diffusion gap, network density) we get mechanistic
insight and a screening tool. *Code:* generate labelled ensemble via `scan_k.classify`,
fit a simple model. *Needs:* ensemble harness + a feature extractor.

---

## Theme C — From linear theory to nonlinear pattern

**C1. Does the linear `k*` predict the realized pattern wavelength?**
Empirically `k*≈12.8` (`L*≈0.49`). *Rationale:* the central validation that linear theory
is predictive in this model. *Code:* `scan_k.max_k` gives `k*`; measure the dominant
wavelength from the PDE solution's spatial power spectrum (FFT of the biomass field).
Notebook 02 already studies length scales and notebook 09 sizes `dx` to `L*`. Directly
answerable; needs a small spectral-peak extractor.

**C2. What sets pattern amplitude, and is the bifurcation super- or sub-critical?**
Max `Re λ ≈ 0.07` is near-marginal/slow. *Rationale:* supercritical → smooth amplitude
growth and good linear prediction; subcritical → hysteresis, hard onset, possible spatial
localized states. *Code:* sweep a control parameter (`K` or `l`) across the
`fr_instab_line` boundary and record steady pattern amplitude from PDE runs. *Needs:*
amplitude metric + ideally a weakly-nonlinear amplitude-equation derivation (new analytic
work) to predict the branch direction.

**C3. 1D vs 2D morphology — stripes, spots, or labyrinths?**
*Rationale:* 2D selects between stripe/spot/hexagon patterns depending on the quadratic
nonlinearity; the Modified MiCRM has products `N_i R_a`, so quadratic terms are present.
*Code:* `solve_pde_2d_imex` (notebook 09) produces 2D fields; `solve_pde` for 1D. *Needs:*
morphology classifier (structure factor anisotropy, spot detection) — modest.

**C4. How fast does the pattern form, and does the near-marginal growth rate matter
experimentally?** *Rationale:* `Re λ≈0.07` implies slow onset; the relevant question is
whether patterns can establish within a realistic colony-growth window. *Code:* time-to-
pattern from PDE runs vs `1/Re λ` from `scan_k`. Directly answerable with current solvers.

---

## Theme D — Robustness and richer dynamics

**D1. Boundary conditions: periodic vs closed (zero-flux).**
*Rationale:* real colonies have boundaries; mode selection and defect placement differ.
*Code:* `bc="periodic"|"closed"` supported in `space.py` and `solve_pde`/`solve_pde_2d`
(BDF). Note `solve_pde_2d_imex` is **periodic-only** (FFT), so closed-BC long 2D runs need
the BDF path or a new IMEX variant. Mostly answerable now.

**D2. Domain-size / finite-size effects and mode quantization.**
*Rationale:* small domains quantize allowed `k`, so the realized wavelength jumps with box
size; tests robustness of the `k*` prediction. *Code:* vary grid size / `dx` in PDE runs;
compare to `scan_k`. Directly answerable.

**D3. Multistability and hysteresis under parameter cycling.**
*Rationale:* the minimal model already has two coexistence branches
(`mm_get_nospace_sol` returns up to two), suggesting bistability; does the spatial system
inherit hysteresis? *Code:* minimal-model branches exist; initialize PDE runs from
different states and cycle a parameter. *Needs:* parameter-continuation / branch-tracking
capability (new) for a clean hysteresis loop; can be approximated by careful re-runs.

**D4. Robustness to noise (demographic/environmental).**
*Rationale:* tests whether near-marginal patterns survive stochasticity — important given
small `Re λ`. *Code:* current solvers are deterministic. *Needs:* a stochastic forcing
term (additive noise in the IMEX stepper is easy; multiplicative/demographic noise is more
work) — new capability, but a small extension of `solve_pde_2d_imex`.

**D5. IMEX vs BDF agreement and numerical reliability.**
*Rationale:* a methods-credibility check given near-marginal growth where discretization
choices can flip stability. *Code:* `diffusion_symbol_2d` deliberately uses the *discrete*
FD symbol so IMEX matches the BDF/FD dispersion; cross-check `solve_pde_2d_imex` vs
`solve_pde_2d` on a small grid. Directly answerable now — good supplementary figure.

---

## Theme E — Statistical ensembles and scaling

**E1. Distribution of `k*` (and `L*`) across random communities.**
*Rationale:* is there a characteristic, predictable length scale for cross-feeding
patterns, or is it broadly distributed? *Code:* ensemble of `scan_k.max_k`. *Needs:*
ensemble harness only — linear-stability is cheap so this is a large, easy result.

**E2. Fraction of communities that are Turing-unstable as a function of sampler knobs.**
*Rationale:* a phase diagram in `(K, l, p, num_byproducts, prob_eating)` of instability
probability. *Code:* `classify` over sampler grids; notebooks 01/03 already do
phase-diagram-style work. *Needs:* ensemble sweep harness; otherwise ready.

**E3. Seed dependence and self-averaging.**
*Rationale:* current empirical numbers are single-community/seed-dependent; the paper needs
ensemble statistics to make general claims. *Code:* `JansSampler3.sample(rng)` is seedable.
*Needs:* the same ensemble harness (run many seeds, aggregate).

---

## Theme F — Experimental connection and predictions

**F1. Mapping to a real cross-feeding consortium (e.g. acetate/lactate cross-feeders).**
*Rationale:* grounds the model; lets parameters be set from data rather than sampled.
*Code:* `MiCRMParams` accepts arbitrary `c, l, D, K`; minimal model is a clean 1-strain
caricature of a single cross-feeding step. *Needs:* a parameterization mapping from
literature kinetics — primarily a modelling/lit task, not code.

**F2. Testable prediction: pattern wavelength scales with the diffusion gap.**
*Rationale:* `analytics.fr2_km2` predicts `k*²` as an explicit function of `(l, p)`;
yields a falsifiable scaling `L* ∝ √(D_R/r)`-type law. *Code:* `fr2_km2_vec` +
`ksquared_to_L` already give this. Directly answerable and a strong, concrete prediction.

**F3. Prediction: knocking out a byproduct route abolishes the pattern.**
*Rationale:* a clean experimental intervention (deleting a cross-feeding link). *Code:*
zero out the relevant `D[i,a,b]` (or a strain's consumption) and re-run `scan_k` /
PDE to show loss of instability. Directly answerable now.

---

## Highest-leverage first paper (a coherent story)

A tight, mostly-already-feasible arc that establishes the mechanism, validates it, and
makes a testable prediction:

1. **A1 + A2 (mechanism).** Show that differential diffusion (fast influx resource vs
   sessile strains) plus leakage-driven cross-feeding is necessary and sufficient for
   Turing instability, with the minimal-model analytics (`analytics.py`) giving the
   closed-form band and the full model (`scan_k`) confirming it. *Fully feasible now.*

2. **A4 / E2 (the single-influx setting and its phase diagram).** Establish single-influx
   cross-feeding as the clean canonical setup and map instability probability across
   `(K, l, p)`. *Needs only an ensemble-sweep harness — the one piece of infrastructure
   worth building first.*

3. **C1 + F2 (linear theory is predictive).** Demonstrate that the linear `k*` predicts
   the realized PDE wavelength (1D and 2D via `solve_pde` / `solve_pde_2d_imex`,
   notebooks 02/09) and that `L*` follows the analytic `fr2_km2` scaling with the
   diffusion gap. *Feasible now; needs a spectral-peak extractor.*

4. **B1 (the ecological payoff).** Show that pattern formation changes the coexisting set
   relative to the well-mixed fixed point (rescue or exclusion), using `reduce.prune_strains`
   plus a re-invasion test on the patterned state. *Needs long IMEX 2D runs + a small
   re-invasion driver — this is the novel ecological result that lifts the paper above a
   pure pattern-formation study.*

5. **F3 (a falsifiable experimental hook).** Close with the byproduct-knockout prediction:
   deleting one cross-feeding link abolishes the pattern (`scan_k` ± PDE). *Fully feasible
   now and cheap.*

This arc is buildable almost entirely with existing code; the only genuinely new
infrastructure on the critical path is (i) an **ensemble-sweep/seed harness** and (ii) a
**re-invasion test + spectral/morphology metrics** — both small. Everything in Themes D
(noise, hysteresis/continuation) and the 2–3 species semi-symbolic reduction (A5, C2) is
better deferred to a second paper, as each needs real new capability (stochastic PDE,
parameter continuation, or fresh analytics).
