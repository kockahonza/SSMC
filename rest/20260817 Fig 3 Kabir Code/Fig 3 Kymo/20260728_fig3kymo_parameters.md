# Fig 3 Kymo — parameters

Parameters of the kymograph runs in this folder, written by `20260728_run_fig3kymo.py`
alongside `20260728_fig3kymo_data.npz`.

## Model

The Modified MiCRM: `Ns` strains `N` consume `Nr` resources `R` and excrete
byproducts, on a periodic one-dimensional domain.

$$\frac{\partial N_i}{\partial t} = g_i N_i \left( \sum_a w_a (1 - l_{ia}) c_{ia} R_a - m_i \right) + D_{N_i} \frac{\partial^2 N_i}{\partial x^2}$$

$$\frac{\partial R_a}{\partial t} = K_a - r_a R_a - \sum_i N_i c_{ia} R_a + \sum_i \sum_b D_{iab} \frac{w_b}{w_a} l_{ib} N_i c_{ib} R_b + D_{R_a} \frac{\partial^2 R_a}{\partial x^2}$$

Exactly one resource is supplied externally ("single influx"); the strains live
off it and off each other's byproducts. The control parameter of this folder is
the ratio of the byproduct diffusion coefficient to that of the influx resource,

$$p = \frac{D_{\rm byprod}}{D_{\rm ext}}.$$

## Community

One community is drawn and re-used at every `p`, with only its diffusion
coefficients rebuilt between runs, so the kymographs differ by the ratio alone.

| Symbol | Meaning | Value in this run |
| --- | --- | --- |
| `Ns` | number of strains | 10 |
| `Nr` | number of resources | 10 |
| `K` | influx into the single supplied resource | 10 |
| `l_influx` | leakage of the influx resource | 0.999 |
| `num_byproducts` | resources each consumed resource is routed into | 2 |
| `prob_eating` | chance a strain consumes a given byproduct resource | 0.7 |
| `prob_eating_influx` | chance a strain consumes the influx resource | 1 |
| `g_i` | growth efficiency | 1 |
| `w_a` | resource energy density | 1 |
| `r_a` | resource dilution rate | 1 |
| `m_i` | strain death rate | lognormal, `s=0.3`, `scale=1.0` |
| `c_ia` | consumption rate (byproducts) | lognormal, `s=0.3`, `scale=1.0` |
| `c_ia` | consumption rate (influx resource) | lognormal, `s=0.3`, `scale=1.0` |
| `l_ia` | leakage fraction (byproducts) | uniform on (0.3, 0.8) |
| `D_iab` | byproduct routing weights | 2 resources drawn without replacement, weights ~ U(0,1) normalised |
| `D_N` | strain motility | 1e-06 |
| `D_ext` | influx-resource diffusion | 1 |
| `D_byprod` | byproduct diffusion | `p * D_ext`, see below |

## Acceptance criteria

Communities were drawn until one satisfied all of:

1. the well-mixed system reached a steady state by `t = 2000`
   (`rtol = 1e-07`, `atol = 1e-09`, accepted when
   `max|du/dt| < 1e-08`) without the strains going extinct;
2. at least `min_survivors = 0` strains remained above
   `1e-06` at that fixed point — this draw kept
   **2** of 10;
3. the fixed point was Turing-unstable at *every* `p`: stable when well mixed,
   with a positive growth rate at some `k > 0`, over
   `k` in (0.001, 500) sampled at 2000 points.

The accepted community was draw number **1**.

## Runs

Each run starts from the homogeneous fixed point perturbed multiplicatively by
Gaussian noise of amplitude 0.001, and is integrated with the stiff BDF
solver (`rtol = 1e-06`, `atol = 1e-09`) using a
block-tridiagonal Jacobian sparsity pattern. Space is discretised with a second-order central difference on
a periodic grid of 1024 cells.

| Symbol | Meaning | Value in this run |
| --- | --- | --- |
| `t_max` | final time | 1e+06 |
| `ssize` | grid cells along the domain | 1024 |
| `n_peaks` | box length in dominant wavelengths | 20 |
| `L` | box length, `n_peaks * 2*pi/k*` | set per `p`, see below |
| `seed` | master seed for the community and the noise | 20260728 |
| recording times | merged linear and log grid | 599 rows to `t_max` |

The box is sized separately at each `p` so that it starts out able to hold
`n_peaks` peaks of that ratio's fastest-growing wavelength. Since `k*` falls as
the byproducts speed up, the domain grows with `p`.

| `p` | `D_byprod` | `k*` | `lambda* = 2*pi/k*` | growth rate | `L` | `dx` |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 0.01 | 11.507 | 0.546 | 0.1281 | 10.921 | 0.01066 |
| 0.1 | 0.1 | 7.755 | 0.810 | 0.05295 | 16.205 | 0.01582 |
| 1 | 1 | 6.504 | 0.966 | 0.009011 | 19.320 | 0.01887 |

## Seeding

The community is drawn from `SeedSequence([seed, 0])` and each run's spatial
noise from `SeedSequence([seed, 1, round(p * 1e9)])`, so the community, the
grid and the ratio are independent: changing `ssize` does not change which
community is drawn, and each `p` gets its own perturbation.
