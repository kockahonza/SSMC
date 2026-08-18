# SSMC-python

Minimal Python port of the core SSMC (Spatial Stochastic Microbial Communities) codebase. Reproduces the core workflow from `SSMC-main/notebooks/forfigures/`: random community sampling, ODE steady-state solving, linear stability analysis, and 1D PDE evolution of the Modified MiCRM model.

## Scope

- **Modified MiCRM** — strains consume resources and excrete byproducts
- **Minimal Model** — 1-strain 2-resource analytical reduction with closed-form steady states
- **Random sampling** — `JansSampler3` (one of several samplers from the Julia codebase)
- **Linear stability analysis** — eigenvalue scan over wavenumbers with Turing-instability classification
- **1D PDE** — periodic and closed boundary conditions, second-order finite differences
- **Analytical phase boundaries** — closed-form extinction and instability curves in (K, l) space

## Install

```bash
pip install -e .
# with dev extras (pytest, jupyter):
pip install -e ".[dev]"
```

## Layout

```
ssmc/
├── params.py          # MiCRMParams, SpatialMiCRMParams dataclasses
├── minimal_model.py   # MMParams + closed-form steady states
├── rhs.py             # numba-JIT'd ODE RHS (non-spatial and spatial)
├── space.py           # 1D finite-difference diffusion
├── linstab.py         # Jacobian construction + wavenumber scan
├── solver.py          # solve_steady_state, solve_pde (scipy.integrate wrappers)
├── sampling.py        # JansSampler3 random community sampler
├── analytics.py       # fr_ext_line_K, fr_cor1_instab_line_K, fr2_km2, ...
└── plotting.py        # matplotlib helpers for phase diagrams and dispersion

notebooks/
├── 20260414_01_mm_phase_diagram.ipynb
├── 20260414_02_mm_pde_length_scales.ipynb
└── 20260414_03_single_influx.ipynb

tests/
├── test_minimal_model.py
├── test_linstab.py
└── test_analytics.py
```

## Quick example

```python
import numpy as np
from ssmc.minimal_model import MMParams, mm_get_nospace_sol, mmp_to_smicrm
from ssmc.linstab import scan_k, classify
from ssmc.solver import solve_steady_state

mmp = MMParams(K=10.0, m=1.0, c=1.0, l=0.8)
sols = mm_get_nospace_sol(mmp)
print(sols)

sparams = mmp_to_smicrm(mmp, DN=1e-12, DI=1.0, DR=0.01)
u0 = sols[0]
u_ss, converged = solve_steady_state(sparams.micrm, u0)

ks = np.linspace(0.01, 5, 200)
result = scan_k(sparams, u_ss, ks)
print(classify(result), "code =", result.code)
```

## Reference

This is a port of the Julia implementation in `../SSMC-main/`. See the Julia source for the original (more complete) implementation and `../SSMC-main/docs/20260414_python_port_plan.md` for the port design.
