"""Scan byproduct mobility ``p = D_x/D_ext`` with replicates, recording peaks vs time.

Six mobilities (0.05, 0.1, 0.25, 0.5, 1, 2) times ten noise seeds are run out to
``t = 1e8``, each on its own periodic box sized to hold 100 pattern wavelengths. Every
run records only the number of peaks, on a logarithmic time grid; the spatial profiles
are counted and discarded inside the model module, so the whole scan fits in one small
npz.

The runs are independent single-threaded stiff solves, so they are farmed out one per
core. Results are checkpointed to the npz as they land, and the two figures can be
redrawn from it later with ``--plot-only``.

Outputs, written next to this script:

* ``20260727_fig3scan_peaks.png`` / ``.svg`` -- peak count against time, one band per
  ``p`` (replicate mean with min-max spread);
* ``20260727_fig3scan_final.png`` / ``.svg`` -- peak count reached at ``t_max`` against
  ``p``, with the linear-instability prediction ``L / (2 pi / k*)`` for reference;
* ``20260727_fig3scan_peaks.npz`` -- the peak traces and run metadata (see
  :func:`save_npz` for the layout).

All model code lives in ``20260727_fig3scan_functions.py``, loaded by path because its
name starts with a digit and sits in a folder with a space in its name.

The full scan is a cluster job (see ``20260727_run_fig3scan.sbatch``)::

    python 20260727_run_fig3scan.py

A quick check of the machinery costs about a minute::

    python 20260727_run_fig3scan.py --t-max 1e3 --replicates 2 --ssize 512
"""

# Cap BLAS/OpenMP threads to 1 *before* numpy is imported. Every worker is one stiff
# solve on a 3-component system; parallelism comes from running many of them at once, so
# BLAS threads inside a worker would only fight the other workers for cores.
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
    "fig3scan_functions", _HERE / "20260727_fig3scan_functions.py"
)
fig3scan = importlib.util.module_from_spec(_spec)
sys.modules["fig3scan_functions"] = fig3scan  # register so forked workers can import it
_spec.loader.exec_module(fig3scan)

DATE = "20260727"
STEM = f"{DATE}_fig3scan"

# The mobilities scanned. Below p ~ 0.05 the peak count stays locked at its
# linear-instability value; above p ~ 2.2 the instability itself disappears.
P_VALUES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)

sns.set_context("talk", rc={"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15})
sns.set_style("whitegrid", {"grid.color": ".9", "grid.linestyle": "--",
                            "axes.edgecolor": ".6", "xtick.bottom": True,
                            "ytick.left": True})


def run_one(task):
    """Worker: solve one ``(p, replicate)`` system and return its peak trace."""
    i, rep, p, seed, kwargs = task
    t0 = time.perf_counter()
    trace = fig3scan.run_peaks(p=p, seed=seed, verbose=False, **kwargs)
    note = "" if trace.success else f"  SOLVER FAILED: {trace.message}"
    print(f"  p={p:<7g} seed={seed:<12d} {time.perf_counter() - t0:8.1f}s  "
          f"k*={trace.k_star:6.3f}  peaks {trace.n_peaks_target} -> "
          f"{trace.final_peaks}{note}", flush=True)
    return i, rep, trace


def save_npz(path: pathlib.Path, ps, seeds, t_grid, traces, args) -> None:
    """Write the scan to disk, tolerating runs that have not finished yet.

    Called after every completed run (about a millisecond) so the file on disk is always
    a usable snapshot: a scan that overruns its wall limit still leaves every finished
    replicate behind.

    Layout, with ``n_p`` mobilities and ``n_rep`` replicates:

    ``p`` (n_p), ``seeds`` (n_p, n_rep), ``t`` (n_times)
        what was run, and the recording grid shared by every run.
    ``peaks`` (n_p, n_rep, n_times)
        the result. NaN marks a replicate that has not run; a zero is a real measurement
        meaning the field was still too flat to hold peaks. See
        ``fig3scan_functions.formed_peaks``.
    ``done``, ``success`` (n_p, n_rep)
        which slots have run at all, and which reached ``t_max``.
    ``k_star``, ``max_growth``, ``L_box`` (n_p)
        linear stability and box length. These follow from ``p`` alone, so replicates of
        one ``p`` share them and they carry no replicate axis.
    ``t_max``, ``ssize``, ``n_peaks``, ``K``
        scalars recording the configuration.
    """
    shape = (len(ps), seeds.shape[1])
    peaks = np.full(shape + (t_grid.size,), np.nan)
    scalars = {name: np.full(len(ps), np.nan)
               for name in ("k_star", "max_growth", "L_box")}
    done = np.zeros(shape, dtype=bool)
    success = np.zeros(shape, dtype=bool)

    for (i, rep), trace in traces.items():
        peaks[i, rep, :trace.peaks.size] = trace.peaks
        for name in scalars:
            scalars[name][i] = getattr(trace, name)
        done[i, rep] = True
        success[i, rep] = trace.success

    np.savez(path, p=np.asarray(ps), seeds=seeds, t=t_grid, peaks=peaks,
             done=done, success=success, t_max=args.t_max, ssize=args.ssize,
             n_peaks=args.n_peaks, K=args.K, **scalars)


def report(npz_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Draw both figures and print the summary table from a saved scan."""
    with np.load(npz_path) as data:
        draw_figures(data, out_dir)
        print_summary(data)


def draw_figures(data, out_dir: pathlib.Path) -> None:
    """Draw both figures from the contents of the npz (or an equivalent mapping)."""
    ps, t, peaks = data["p"], data["t"], data["peaks"]
    n_peaks_target = int(data["n_peaks"])
    t_max, n_reps = float(data["t_max"]), peaks.shape[1]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig3scan.plot_peaks_vs_time(ax, t, peaks, ps, n_peaks_target=n_peaks_target)
    ax.legend(fontsize=11, ncol=2, frameon=False)
    ax.set_title(f"coarsening vs byproduct mobility  "
                 f"({n_reps} replicates, $t_{{\\max}}$={t_max:g})")
    _save(fig, out_dir / f"{STEM}_peaks")

    fig, ax = plt.subplots(figsize=(8, 6))
    fig3scan.plot_final_peaks(ax, ps, peaks[:, :, -1], n_peaks_target=n_peaks_target)
    ax.legend(fontsize=12, frameon=False)
    ax.set_title(f"peaks at $t$={t_max:g}")
    _save(fig, out_dir / f"{STEM}_final")


def _save(fig, path_stem: pathlib.Path) -> None:
    fig.tight_layout()
    for suffix in (".png", ".svg"):
        fig.savefig(path_stem.with_suffix(suffix), dpi=150)
    plt.close(fig)
    print(f"wrote {path_stem}.png and {path_stem}.svg", flush=True)


def print_summary(data) -> None:
    """One line per ``p``: its linear stability and the peak count it settles at.

    A zero count means the run finished with no pattern -- either the instability is too
    slow to have grown by ``t_max``, or the parameters are not Turing-unstable at that
    ``p`` -- so it is reported as such rather than averaged in as a count of zero peaks.
    """
    final = fig3scan.formed_peaks(data["peaks"][:, :, -1])
    n_run = data["done"].sum(axis=1)

    print(f"\n{'p':>7} {'k*':>7} {'growth':>9}   peaks at t_max (mean [min, max])")
    for i, p in enumerate(data["p"]):
        formed = final[i][np.isfinite(final[i])]
        if n_run[i] == 0:
            stats = "(no runs)"
        elif formed.size == 0:
            stats = "(no pattern in any replicate)"
        else:
            stats = f"{formed.mean():6.1f} [{formed.min():.0f}, {formed.max():.0f}]"
            if formed.size < n_run[i]:
                stats += f"  ({n_run[i] - formed.size} with no pattern)"
        print(f"{p:>7g} {data['k_star'][i]:>7.3f} {data['max_growth'][i]:>+9.4f}   "
              f"{stats}")


def n_workers(requested) -> int:
    """Worker count: what was asked for, else the cores this process may actually use.

    ``sched_getaffinity`` respects the cgroup and taskset limits a scheduler imposes,
    which ``cpu_count`` does not, so it covers the SLURM allocation and a shared
    workstation alike.
    """
    if requested is not None:
        return requested
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ps", type=float, nargs="+", default=list(P_VALUES),
                    help="byproduct mobilities D_x/D_ext to scan (default: %(default)s)")
    ap.add_argument("--replicates", type=int, default=10,
                    help="independent noise seeds per p (default: %(default)s)")
    ap.add_argument("--t-max", type=float, default=1e8,
                    help="final time (default: %(default)g)")
    ap.add_argument("--ssize", type=int, default=2048,
                    help="grid cells along the domain; 2048 gives ~20 points per "
                         "wavelength at 100 peaks (default: %(default)s)")
    ap.add_argument("--n-peaks", type=int, default=100,
                    help="wavelengths each box is sized to hold (default: %(default)s)")
    ap.add_argument("--n-times", type=int, default=300,
                    help="log-spaced times recorded per run (default: %(default)s)")
    ap.add_argument("--K", type=float, default=fig3scan.K_INFLUX,
                    help="external supply of r_ext (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=20260727,
                    help="master seed; every run's noise is derived from it "
                         "(default: %(default)s)")
    ap.add_argument("--n-procs", type=int, default=None,
                    help="worker processes (default: the cores this job may use)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=_HERE,
                    help="where to write the outputs (default: this folder)")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figures from an existing npz without solving")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{STEM}_peaks.npz"

    if args.plot_only:
        report(out, args.out_dir)
        return

    ps = [float(p) for p in args.ps]
    # One independent seed per (p, replicate) slot, all derived from the master seed so
    # the whole scan is reproducible from a single number, and all saved so any run can
    # be reproduced on its own.
    seeds = np.random.SeedSequence(args.seed).generate_state(
        len(ps) * args.replicates).reshape(len(ps), args.replicates)
    # run_peaks builds its own recording grid from t_max and n_times; log_times is a pure
    # function of those, so this copy for the npz is the identical grid.
    t_grid = fig3scan.log_times(args.t_max, n_times=args.n_times)

    kwargs = dict(t_max=args.t_max, n_times=args.n_times, ssize=args.ssize,
                  n_peaks=args.n_peaks, K=args.K)
    # Tasks carry their (p, replicate) slot so results can be filed as they arrive out of
    # order: runs near p = 1 coarsen the most and take far longer than the rest.
    tasks = [(i, rep, p, int(seeds[i, rep]), kwargs)
             for i, p in enumerate(ps) for rep in range(args.replicates)]

    # Size the pool to the allocation so we do not spill onto cores the scheduler did not
    # give us.
    n_procs = max(1, min(n_workers(args.n_procs), len(tasks)))

    print(f"{len(ps)} values of p x {args.replicates} replicate(s) = {len(tasks)} runs "
          f"on {n_procs} process(es); t_max={args.t_max:g}, ssize={args.ssize}, "
          f"n_peaks={args.n_peaks}, K={args.K:g}", flush=True)
    print(f"p = {', '.join(f'{p:g}' for p in ps)}", flush=True)

    traces, t0 = {}, time.perf_counter()
    with mp.Pool(n_procs) as pool:
        for n_done, (i, rep, trace) in enumerate(
                pool.imap_unordered(run_one, tasks), start=1):
            traces[(i, rep)] = trace
            save_npz(out, ps, seeds, t_grid, traces, args)
            if n_done < len(tasks):
                print(f"    [{n_done}/{len(tasks)} done, checkpointed]", flush=True)

    print(f"\nscan finished in {time.perf_counter() - t0:.1f}s; wrote {out}", flush=True)
    report(out, args.out_dir)


if __name__ == "__main__":
    main()
