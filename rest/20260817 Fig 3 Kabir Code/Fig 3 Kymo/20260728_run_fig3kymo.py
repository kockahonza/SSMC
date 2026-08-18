"""Kymographs of one single-influx community across the diffusion ratio p = D_byprod/D_ext.

A single Turing-unstable community is drawn once and then evolved separately at
each ``p`` in {0.01, 0.1, 1}, on a periodic box sized to hold twenty peaks of
that ratio's fastest-growing wavelength, out to ``t = 1e6``. Only the diffusion
coefficients differ between the runs, so the kymographs may be read against each
other directly.

Outputs, with ``STEM = 20260728_fig3kymo``:

* ``STEM_data.npz`` -- the recorded biomass fields and the run metadata, enough
  to redraw every figure without integrating again (see :func:`save_npz` for the
  layout, and the companion notebook for a reload);
* ``STEM_parameters.md`` -- every parameter of the run, plus what the linear
  stability analysis says about each ``p``;
* ``Panels/STEM_kymograph_{linear,log}.{png,svg}`` -- all ratios side by side,
  once with a linear time axis and once logarithmic;
* ``Panels/STEM_p{p}_{linear,log}.{png,svg}`` -- the same runs one panel at a
  time, for use as standalone figures.

The two time axes show different things: the log axis spreads out the early
decades in which the instability grows out of the noise and selects its
wavelength, the linear axis shows the late state and any slow coarsening. Both
are drawn from one integration, recorded on a merged linear-and-log time grid.

All model code lives in ``20260728_fig3kymo_functions.py``, loaded by path
because its name starts with a digit and sits in a folder with a space in its
name.

Run with defaults::

    python 20260728_run_fig3kymo.py

Redraw the figures and the parameter file from saved data::

    python 20260728_run_fig3kymo.py --plot-only

Ask for a richer community, or a different one::

    python 20260728_run_fig3kymo.py --min-survivors 4 --seed 7
"""

# Cap BLAS/OpenMP threads to 1 *before* numpy is imported. Each run is one stiff
# solve, which BLAS does not usefully parallelise; letting every worker spawn
# threads only adds contention.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import importlib.util
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: we only save figures, never show them
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load the essential-functions module by path --------------------------- #
_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "fig3kymo_functions", _HERE / "20260728_fig3kymo_functions.py"
)
fig3kymo = importlib.util.module_from_spec(_spec)
sys.modules["fig3kymo_functions"] = fig3kymo  # register so forked workers can import it
_spec.loader.exec_module(fig3kymo)

DATE = "20260728"
STEM = f"{DATE}_fig3kymo"
DRIVER = f"{DATE}_run_fig3kymo.py"

# The two views every run is drawn in, as (log time?, filename tag).
TIME_AXES = ((False, "linear"), (True, "log"))

# The model, for the generated parameter file. Kept out of the f-string that
# builds that file so the LaTeX braces need no doubling.
MODEL_EQUATIONS = r"""$$\frac{\partial N_i}{\partial t} = g_i N_i \left( \sum_a w_a (1 - l_{ia}) c_{ia} R_a - m_i \right) + D_{N_i} \frac{\partial^2 N_i}{\partial x^2}$$

$$\frac{\partial R_a}{\partial t} = K_a - r_a R_a - \sum_i N_i c_{ia} R_a + \sum_i \sum_b D_{iab} \frac{w_b}{w_a} l_{ib} N_i c_{ib} R_b + D_{R_a} \frac{\partial^2 R_a}{\partial x^2}$$"""

RATIO_DEFINITION = r"$$p = \frac{D_{\rm byprod}}{D_{\rm ext}}.$$"

sns.set_context("talk", rc={"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15})
sns.set_style("whitegrid", {"grid.color": ".9", "grid.linestyle": "--",
                            "axes.edgecolor": ".6", "xtick.bottom": True,
                            "ytick.left": True})


# --------------------------------------------------------------------------- #
# Running                                                                     #
# --------------------------------------------------------------------------- #


def run_one(task):
    """Evolve the shared community at one ``p``. Runs in a worker process."""
    index, community, p, kwargs = task
    t0 = time.perf_counter()
    run = fig3kymo.run_at_p(community, p, **kwargs)
    print(f"  p={p:<8g} {time.perf_counter() - t0:7.1f}s  k*={run.k_star:6.3f}  "
          f"L={run.L_box:7.2f}  {len(run.t)} rows"
          f"{'' if run.success else '  SOLVER FAILED: ' + run.message}", flush=True)
    return index, run


def save_npz(path: pathlib.Path, ps, t_grid, runs: dict, community, args) -> None:
    """Write everything the figures and the notebook need.

    Layout, with ``n_p`` ratios and ``n_times`` recording times::

        p (n_p)                        byproduct/influx diffusion ratios
        t (n_times)                    recording grid, shared by every run
        x (n_p, ssize)                 grid positions; differs with p because L does
        biomass (n_p, n_times, ssize)  total strain biomass; NaN where not recorded
        done, success (n_p)            run finished / solver reached t_max
        k_star, max_mrl, L_box (n_p)   linear stability and box size at each p
        t_max, ssize, n_peaks, K, l_influx, Ns, Nr, seed,
        min_survivors, n_survivors, n_draws                          scalars

    Called after every completed run, so the file on disk is always a usable
    snapshot of the runs that have finished.
    """
    n_p, n_times = len(ps), len(t_grid)
    x = np.full((n_p, args.ssize), np.nan)
    biomass = np.full((n_p, n_times, args.ssize), np.nan)
    k_star = np.full(n_p, np.nan)
    max_mrl = np.full(n_p, np.nan)
    L_box = np.full(n_p, np.nan)
    done = np.zeros(n_p, dtype=bool)
    success = np.zeros(n_p, dtype=bool)

    for i, run in runs.items():
        x[i] = run.x
        biomass[i, :len(run.t)] = run.biomass   # a failed solve stops early
        k_star[i], max_mrl[i], L_box[i] = run.k_star, run.max_mrl, run.L_box
        done[i], success[i] = True, run.success

    np.savez(path, p=np.asarray(ps, dtype=float), t=t_grid, x=x, biomass=biomass,
             done=done, success=success, k_star=k_star, max_mrl=max_mrl, L_box=L_box,
             t_max=args.t_max, ssize=args.ssize, n_peaks=args.n_peaks, K=args.K,
             l_influx=args.l_influx, Ns=args.Ns, Nr=args.Nr, seed=args.seed,
             min_survivors=args.min_survivors, n_survivors=community.n_survivors,
             n_draws=community.n_draws)


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #


def _save(fig, path_stem: pathlib.Path) -> None:
    """Write one figure as both png and svg.

    The extension is appended rather than set with ``with_suffix``, which would
    read the decimal point of a stem like ``..._p0.01_linear`` as the start of
    an extension and truncate the name.
    """
    fig.tight_layout()
    for suffix in (".png", ".svg"):
        fig.savefig(path_stem.with_name(path_stem.name + suffix), dpi=150)
    plt.close(fig)
    print(f"wrote {path_stem}.png and {path_stem}.svg", flush=True)


def _panel(ax, data, i: int, log_time: bool):
    """Draw run ``i`` of a loaded npz onto ``ax`` and return the mesh."""
    return fig3kymo.plot_kymograph(ax, data["x"][i], data["t"], data["biomass"][i],
                                   log_time=log_time)


def draw_comparison(data, panel_dir: pathlib.Path) -> None:
    """One row of panels per time axis: every ratio side by side.

    Each panel keeps its own colour scale, since the biomass a pattern settles
    at changes with ``p`` and a shared scale would flatten the smaller ones.
    """
    ps, done = data["p"], data["done"]
    shown = np.flatnonzero(done)
    if shown.size == 0:
        return

    for log_time, name in TIME_AXES:
        fig, axes = plt.subplots(1, shown.size, figsize=(6.0 * shown.size, 5.0),
                                 squeeze=False)
        for ax, i in zip(axes[0], shown):
            im = _panel(ax, data, i, log_time)
            ax.set_title(f"$p$={ps[i]:g}  |  $k^*$={data['k_star'][i]:.2f}, "
                         f"$L$={data['L_box'][i]:.1f}")
            fig.colorbar(im, ax=ax, shrink=0.8, label="total biomass")
        _save(fig, panel_dir / f"{STEM}_kymograph_{name}")


def draw_singles(data, panel_dir: pathlib.Path) -> None:
    """One standalone figure per ratio per time axis."""
    ps = data["p"]
    for i in np.flatnonzero(data["done"]):
        for log_time, name in TIME_AXES:
            fig, ax = plt.subplots(figsize=(15, 5))
            im = _panel(ax, data, i, log_time)
            ax.set_title(f"total biomass  |  $p=D_{{\\rm byprod}}/D_{{\\rm ext}}$={ps[i]:g}, "
                         f"$k^*$={data['k_star'][i]:.2f}, $L$={data['L_box'][i]:.2f}")
            fig.colorbar(im, ax=ax, shrink=0.8, label="total biomass")
            _save(fig, panel_dir / f"{STEM}_p{ps[i]:g}_{name}")


# --------------------------------------------------------------------------- #
# Parameter record                                                            #
# --------------------------------------------------------------------------- #


def write_parameters_md(path: pathlib.Path, data) -> None:
    """Write the run's parameters, and what linear stability says about them.

    Everything comes from the saved npz plus the module-level constants, so the
    file can be regenerated from data alone.
    """
    ps, k_star, L_box = data["p"], data["k_star"], data["L_box"]
    ssize = int(data["ssize"])

    rows = []
    for i in range(len(ps)):
        if not data["done"][i]:
            rows.append(f"| {ps[i]:g} | {ps[i] * fig3kymo.D_EXT:g} | not run | | | | |")
            continue
        rows.append(
            f"| {ps[i]:g} | {ps[i] * fig3kymo.D_EXT:g} | {k_star[i]:.3f} | "
            f"{2 * np.pi / k_star[i]:.3f} | {data['max_mrl'][i]:.4g} | "
            f"{L_box[i]:.3f} | {L_box[i] / ssize:.4g} |")

    text = f"""# Fig 3 Kymo — parameters

Parameters of the kymograph runs in this folder, written by `{DRIVER}`
alongside `{STEM}_data.npz`.

## Model

The Modified MiCRM: `Ns` strains `N` consume `Nr` resources `R` and excrete
byproducts, on a periodic one-dimensional domain.

{MODEL_EQUATIONS}

Exactly one resource is supplied externally ("single influx"); the strains live
off it and off each other's byproducts. The control parameter of this folder is
the ratio of the byproduct diffusion coefficient to that of the influx resource,

{RATIO_DEFINITION}

## Community

One community is drawn and re-used at every `p`, with only its diffusion
coefficients rebuilt between runs, so the kymographs differ by the ratio alone.

| Symbol | Meaning | Value in this run |
| --- | --- | --- |
| `Ns` | number of strains | {int(data["Ns"])} |
| `Nr` | number of resources | {int(data["Nr"])} |
| `K` | influx into the single supplied resource | {float(data["K"]):g} |
| `l_influx` | leakage of the influx resource | {float(data["l_influx"]):g} |
| `num_byproducts` | resources each consumed resource is routed into | {fig3kymo.NUM_BYPRODUCTS} |
| `prob_eating` | chance a strain consumes a given byproduct resource | {fig3kymo.PROB_EATING:g} |
| `prob_eating_influx` | chance a strain consumes the influx resource | 1 |
| `g_i` | growth efficiency | 1 |
| `w_a` | resource energy density | 1 |
| `r_a` | resource dilution rate | 1 |
| `m_i` | strain death rate | lognormal, `s=0.3`, `scale=1.0` |
| `c_ia` | consumption rate (byproducts) | lognormal, `s=0.3`, `scale=1.0` |
| `c_ia` | consumption rate (influx resource) | lognormal, `s=0.3`, `scale=1.0` |
| `l_ia` | leakage fraction (byproducts) | uniform on (0.3, 0.8) |
| `D_iab` | byproduct routing weights | {fig3kymo.NUM_BYPRODUCTS} resources drawn without replacement, weights ~ U(0,1) normalised |
| `D_N` | strain motility | {fig3kymo.D_STRAIN:g} |
| `D_ext` | influx-resource diffusion | {fig3kymo.D_EXT:g} |
| `D_byprod` | byproduct diffusion | `p * D_ext`, see below |

## Acceptance criteria

Communities were drawn until one satisfied all of:

1. the well-mixed system reached a steady state by `t = {fig3kymo.SS_T_MAX:g}`
   (`rtol = {fig3kymo.SS_RTOL:g}`, `atol = {fig3kymo.SS_ATOL:g}`, accepted when
   `max|du/dt| < {fig3kymo.SS_TOL:g}`) without the strains going extinct;
2. at least `min_survivors = {int(data["min_survivors"])}` strains remained above
   `{fig3kymo.SURVIVOR_THRESHOLD:g}` at that fixed point — this draw kept
   **{int(data["n_survivors"])}** of {int(data["Ns"])};
3. the fixed point was Turing-unstable at *every* `p`: stable when well mixed,
   with a positive growth rate at some `k > 0`, over
   `k` in ({fig3kymo.KS[0]:g}, {fig3kymo.KS[-1]:g}) sampled at {len(fig3kymo.KS)} points.

The accepted community was draw number **{int(data["n_draws"])}**.

## Runs

Each run starts from the homogeneous fixed point perturbed multiplicatively by
Gaussian noise of amplitude {fig3kymo.NOISE_AMPLITUDE:g}, and is integrated with the stiff BDF
solver (`rtol = {fig3kymo.PDE_RTOL:g}`, `atol = {fig3kymo.PDE_ATOL:g}`) using a
block-tridiagonal Jacobian sparsity pattern. Space is discretised with a second-order central difference on
a periodic grid of {ssize} cells.

| Symbol | Meaning | Value in this run |
| --- | --- | --- |
| `t_max` | final time | {float(data["t_max"]):g} |
| `ssize` | grid cells along the domain | {ssize} |
| `n_peaks` | box length in dominant wavelengths | {float(data["n_peaks"]):g} |
| `L` | box length, `n_peaks * 2*pi/k*` | set per `p`, see below |
| `seed` | master seed for the community and the noise | {int(data["seed"])} |
| recording times | merged linear and log grid | {len(data["t"])} rows to `t_max` |

The box is sized separately at each `p` so that it starts out able to hold
`n_peaks` peaks of that ratio's fastest-growing wavelength. Since `k*` falls as
the byproducts speed up, the domain grows with `p`.

| `p` | `D_byprod` | `k*` | `lambda* = 2*pi/k*` | growth rate | `L` | `dx` |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## Seeding

The community is drawn from `SeedSequence([seed, 0])` and each run's spatial
noise from `SeedSequence([seed, 1, round(p * 1e9)])`, so the community, the
grid and the ratio are independent: changing `ssize` does not change which
community is drawn, and each `p` gets its own perturbation.
"""
    path.write_text(text)
    print(f"wrote {path}", flush=True)


def print_summary(data) -> None:
    """Print what was run, one line per ratio."""
    print(f"\ncommunity: {int(data['n_survivors'])} of {int(data['Ns'])} strains "
          f"survive, found on draw {int(data['n_draws'])}")
    print(f"{'p':>8} {'k*':>8} {'growth':>10} {'L':>9}   solver")
    for i, p in enumerate(data["p"]):
        if not data["done"][i]:
            print(f"{p:>8g} {'':>8} {'':>10} {'':>9}   (not run)")
            continue
        print(f"{p:>8g} {data['k_star'][i]:>8.3f} {data['max_mrl'][i]:>10.4g} "
              f"{data['L_box'][i]:>9.3f}   "
              f"{'reached t_max' if data['success'][i] else 'FAILED'}")


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def n_workers(requested, n_tasks: int) -> int:
    """Worker count: what was asked for, else the cores this process may use."""
    if requested:
        return max(1, min(int(requested), n_tasks))
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:      # not available on every platform
        available = os.cpu_count() or 1
    return max(1, min(available, n_tasks))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ps", type=float, nargs="+", default=list(fig3kymo.P_VALUES),
                    help="byproduct/influx diffusion ratios (default: %(default)s)")
    ap.add_argument("--t-max", type=float, default=1e6,
                    help="final time (default: %(default)g)")
    ap.add_argument("--ssize", type=int, default=1024,
                    help="grid cells along the domain (default: %(default)s)")
    ap.add_argument("--n-peaks", type=float, default=20.0,
                    help="box length in dominant wavelengths (default: %(default)s)")
    ap.add_argument("--Ns", type=int, default=fig3kymo.N_STRAINS,
                    help="number of strains (default: %(default)s)")
    ap.add_argument("--Nr", type=int, default=fig3kymo.N_RESOURCES,
                    help="number of resources (default: %(default)s)")
    ap.add_argument("--min-survivors", type=int, default=0,
                    help="fewest strains allowed to survive at the homogeneous "
                         "fixed point; more than this is fine (default: %(default)s)")
    ap.add_argument("--max-tries", type=int, default=200,
                    help="communities to draw before giving up (default: %(default)s)")
    ap.add_argument("--K", type=float, default=fig3kymo.K_INFLUX,
                    help="energy supplied to the influx resource (default: %(default)s)")
    ap.add_argument("--l-influx", type=float, default=fig3kymo.L_INFLUX,
                    help="leakage of the influx resource (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=20260728,
                    help="master seed for the community and the noise (default: %(default)s)")
    ap.add_argument("--n-procs", type=int, default=None,
                    help="worker processes (default: one per available core)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=_HERE,
                    help="where to write the data and the parameter file "
                         "(default: this folder)")
    ap.add_argument("--panel-dir", type=pathlib.Path, default=None,
                    help="where to write the figures (default: <out-dir>/Panels)")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figures and the parameter file from saved data")
    return ap.parse_args()


def report(npz_path: pathlib.Path, out_dir: pathlib.Path, panel_dir: pathlib.Path) -> None:
    """Redraw every figure and the parameter file from a saved npz."""
    panel_dir.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path) as npz:
        data = dict(npz)   # read the fields once; each figure indexes them repeatedly

    draw_comparison(data, panel_dir)
    draw_singles(data, panel_dir)
    write_parameters_md(out_dir / f"{STEM}_parameters.md", data)
    print_summary(data)


def main() -> None:
    args = parse_args()
    panel_dir = args.panel_dir or args.out_dir / "Panels"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{STEM}_data.npz"

    if args.plot_only:
        report(out, args.out_dir, panel_dir)
        return

    ps = [float(p) for p in args.ps]
    print(f"ps={ps}, t_max={args.t_max:g}, ssize={args.ssize}, "
          f"n_peaks={args.n_peaks:g}, seed={args.seed}", flush=True)

    # One community for every ratio, drawn in the parent so the workers share it.
    t0 = time.perf_counter()
    community = fig3kymo.draw_shared_community(
        ps, np.random.default_rng(np.random.SeedSequence([args.seed, 0])),
        K=args.K, l_influx=args.l_influx, Ns=args.Ns, Nr=args.Nr,
        min_survivors=args.min_survivors, max_tries=args.max_tries,
    )
    if community is None:
        sys.exit(f"no community unstable at every p in {ps} with at least "
                 f"{args.min_survivors} survivors; try another --seed, a larger "
                 f"--max-tries, or fewer --min-survivors")
    print(f"community found in {time.perf_counter() - t0:.1f}s after "
          f"{community.n_draws} draw(s): {community.n_survivors} of {args.Ns} "
          f"strains survive", flush=True)

    t_grid = fig3kymo.kymograph_times(args.t_max)
    kwargs = dict(t_max=args.t_max, seed=args.seed, ssize=args.ssize,
                  n_peaks=args.n_peaks, t_eval=t_grid, verbose=True)
    tasks = [(i, community, p, kwargs) for i, p in enumerate(ps)]

    n_procs = n_workers(args.n_procs, len(tasks))
    print(f"{len(tasks)} runs on {n_procs} process(es), "
          f"~{args.ssize * len(t_grid) * 8 / 1e6:.0f} MB of field per run", flush=True)

    runs = {}
    t0 = time.perf_counter()
    with mp.Pool(n_procs) as pool:
        for index, run in pool.imap_unordered(run_one, tasks):
            runs[index] = run
            save_npz(out, ps, t_grid, runs, community, args)  # always a usable snapshot
            print(f"    [{len(runs)}/{len(tasks)} done, checkpointed]", flush=True)
    print(f"all runs finished in {time.perf_counter() - t0:.1f}s; wrote {out}", flush=True)

    report(out, args.out_dir, panel_dir)


if __name__ == "__main__":
    main()
