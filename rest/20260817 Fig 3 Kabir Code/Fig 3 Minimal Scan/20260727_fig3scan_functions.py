"""Self-contained essentials for scanning byproduct mobility in the minimal model.

The model is the three-variable reduction of the Modified MiCRM used in
``Consolidated/Fig. 3 Minimal``: one strain ``n`` feeding on an externally supplied
resource ``r_ext`` and on the byproduct ``r_x`` that its own metabolism excretes. In one
periodic spatial dimension,

    dn/dt      = n ( c_ext (1 - l) r_ext + c_x r_x - m )  +  D_n   d2n/dx2
    dr_ext/dt  = K - kappa r_ext - c_ext r_ext n          +  D_ext d2r_ext/dx2
    dr_x/dt    = c_ext l r_ext n - kappa r_x - c_x r_x n  +  D_x   d2r_x/dx2

with ``c_ext`` / ``c_x`` the consumption rates of the two resources, ``l`` the fraction
of consumed external resource leaked as byproduct, ``m`` the death rate, ``K`` the
external supply, ``kappa`` the resource dilution rate and ``D_*`` the diffusion
coefficients.

Where ``Fig. 3 Minimal`` explores the model -- kymographs, resource profiles, the window
of usable ``K`` -- this folder answers one question with replicates: how does the number
of peaks evolve as the byproduct mobility ``p = D_x / D_ext`` is varied? The module is
therefore trimmed to that question. A run returns a peak count against time and nothing
else (:class:`PeakTrace`); the spatial profiles are counted and dropped inside
:func:`run_peaks`, so the ~15 MB solver buffer a run allocates does not outlive it and
many long runs can share a node.

The recording grid is logarithmic (:func:`log_times`). Pattern formation and coarsening
are decades apart in time, so even sampling per decade puts the rows where the count
actually changes.

Only numpy, scipy, matplotlib and palettable are needed, so this folder runs without the
``ssmc`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.sparse import diags, kron

from matplotlib.colors import LogNorm
import palettable as pal

# Reaction parameters. The consumption rates are equal and the leakage is total, so the
# external resource contributes no growth directly (c_ext (1 - l) = 0) and all biomass is
# fed through the byproduct. K is the one genuinely free parameter: the coexistence fixed
# point is born in a saddle-node at K = 4 for these values and the Turing instability
# disappears again near K = 8, so K = 5 sits in the middle of the usable window and stays
# unstable over the whole range of p usually scanned.
C_EXT = 1.0        # consumption rate of the external resource
C_X = 1.0          # consumption rate of the byproduct
LEAKAGE = 1.0      # fraction of consumed external resource leaked as byproduct
KAPPA = 1.0        # resource dilution rate
DEATH = 1.0        # strain death rate
K_INFLUX = 5.0     # external supply of r_ext

# Diffusion. The strain is nearly immotile; the external resource sets the unit of
# mobility and p = D_x / D_ext is measured relative to it.
D_N = 1e-3
D_EXT = 1.0

# Wavenumbers scanned when reading off the dispersion relation. The upper limit is
# comfortably past k* for every p in the usable range (k* <= 7 or so).
KS = np.linspace(1e-3, 60.0, 6000)

# Multiplicative perturbation applied to the homogeneous fixed point to seed the pattern.
NOISE_AMPLITUDE = 1e-3

SWEEP_CMAP = pal.matplotlib.Viridis_20.mpl_colormap


# --------------------------------------------------------------------------- #
# Parameters                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MinimalParams:
    """Reaction and diffusion parameters of the three-variable model.

    ``dx`` is the grid spacing of the periodic domain. It plays no part in the fixed
    point or the dispersion relation (which work in continuous ``k``), so it keeps a
    placeholder default and is filled in by :func:`run_peaks` once ``k*`` has fixed the
    box length.
    """

    c_ext: float = C_EXT
    c_x: float = C_X
    l: float = LEAKAGE
    kappa: float = KAPPA
    m: float = DEATH
    K: float = K_INFLUX
    D_n: float = D_N
    D_ext: float = D_EXT
    D_x: float = D_EXT
    dx: float = 1.0

    @classmethod
    def from_p(cls, p: float, D_ext: float = D_EXT, **overrides) -> "MinimalParams":
        """Build a parameter set from the diffusion ratio ``p = D_x / D_ext``."""
        return cls(D_ext=D_ext, D_x=float(p) * D_ext, **overrides)

    @property
    def Ds(self) -> np.ndarray:
        """Diffusion coefficients ordered as the state vector ``(n, r_ext, r_x)``."""
        return np.array([self.D_n, self.D_ext, self.D_x], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Homogeneous fixed point                                                     #
# --------------------------------------------------------------------------- #


def steady_state(prm: MinimalParams) -> np.ndarray:
    """Closed-form coexistence fixed point ``(n, r_ext, r_x)``.

    Eliminating the resources through ``r_ext = K / (kappa + c_ext n)`` and
    ``r_x = c_ext l r_ext n / (kappa + c_x n)`` turns the growth balance into a quadratic
    in the biomass,

        m c_ext c_x n^2 + (m kappa (c_ext + c_x) - K c_ext c_x) n
            + m kappa^2 - K c_ext kappa (1 - l) = 0.

    The two roots are the coexistence state and a saddle; the larger one is the state
    that is stable when well mixed, and is the one returned. A negative discriminant
    means the parameters sit below the saddle-node where the pair is born (``K <= 4`` for
    the defaults) and no coexistence state exists.
    """
    qa = prm.m * prm.c_ext * prm.c_x
    qb = prm.m * prm.kappa * (prm.c_ext + prm.c_x) - prm.K * prm.c_ext * prm.c_x
    qc = prm.m * prm.kappa ** 2 - prm.K * prm.c_ext * prm.kappa * (1.0 - prm.l)

    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        raise ValueError(
            f"no coexistence fixed point for K={prm.K:g} (discriminant {disc:.3g} < 0); "
            "the state is born in a saddle-node at larger K"
        )

    n = (-qb + np.sqrt(disc)) / (2.0 * qa)
    r_ext = prm.K / (prm.kappa + prm.c_ext * n)
    r_x = prm.c_ext * prm.l * r_ext * n / (prm.kappa + prm.c_x * n)
    return np.array([n, r_ext, r_x], dtype=np.float64)


def jacobian(prm: MinimalParams, u: np.ndarray) -> np.ndarray:
    """Well-mixed Jacobian of the three-variable model at state ``u``.

    Written out analytically rather than differenced: it is only 3x3, and the dispersion
    relation needs it at thousands of wavenumbers.
    """
    n, r_ext, r_x = u
    growth = prm.c_ext * (1.0 - prm.l) * r_ext + prm.c_x * r_x - prm.m
    return np.array([
        [growth,                            n * prm.c_ext * (1.0 - prm.l), n * prm.c_x],
        [-prm.c_ext * r_ext,                -prm.kappa - prm.c_ext * n,    0.0],
        [prm.c_ext * prm.l * r_ext - prm.c_x * r_x,
         prm.c_ext * prm.l * n,             -prm.kappa - prm.c_x * n],
    ], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Linear stability                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Dispersion:
    """Summary of how a perturbation of the fixed point grows with its wavenumber.

    Only the three numbers the scan uses are kept; the full growth curve is cheap enough
    to recompute with :func:`dispersion` if it is ever wanted for a plot.
    """

    k_star: float         # wavenumber of the fastest-growing mode
    max_growth: float     # its growth rate
    growth_0: float       # growth rate of the homogeneous (k = 0) mode

    @property
    def is_turing(self) -> bool:
        """True when the well-mixed state is stable but some finite ``k`` grows."""
        return self.growth_0 < 0.0 < self.max_growth

    @property
    def wavelength(self) -> float:
        return 2.0 * np.pi / self.k_star


def dispersion(prm: MinimalParams, ks: Sequence[float] = KS,
               u: Optional[np.ndarray] = None) -> Dispersion:
    """Scan wavenumbers for the growth rate of the fixed point.

    A perturbation ``~ exp(i k x)`` of the homogeneous state grows at the largest real
    eigenvalue of ``J - k^2 diag(Ds)``. A Turing instability shows up as a positive
    maximum at finite ``k`` with the ``k = 0`` mode still decaying. All wavenumbers are
    diagonalised in one stacked call rather than in a Python loop.

    The fastest-growing wavenumber ``k*`` is what sizes the box: knowing it before the
    PDE is touched lets every run start with the same number of peaks, so a later change
    in the count means coarsening rather than a difference in the domain.
    """
    ks = np.ascontiguousarray(ks, dtype=np.float64)
    J = jacobian(prm, steady_state(prm) if u is None else u)

    growth = np.max(np.real(np.linalg.eigvals(
        J - (ks ** 2)[:, None, None] * np.diag(prm.Ds))), axis=1)

    best = int(np.argmax(growth))
    return Dispersion(k_star=float(ks[best]), max_growth=float(growth[best]),
                      growth_0=float(np.max(np.real(np.linalg.eigvals(J)))))


# --------------------------------------------------------------------------- #
# Dynamics                                                                    #
# --------------------------------------------------------------------------- #


def reaction(u: np.ndarray, prm: MinimalParams) -> np.ndarray:
    """Local (space-free) terms for a field of shape ``(3, ssize)``.

    Every grid cell is evaluated at once; with ``ssize = 1`` this is the well-mixed
    right-hand side. Rows are ``(n, r_ext, r_x)``.
    """
    n, r_ext, r_x = u
    return np.stack([
        n * (prm.c_ext * (1.0 - prm.l) * r_ext + prm.c_x * r_x - prm.m),
        prm.K - prm.kappa * r_ext - prm.c_ext * r_ext * n,
        prm.c_ext * prm.l * r_ext * n - prm.kappa * r_x - prm.c_x * r_x * n,
    ])


def rhs_1d(u: np.ndarray, prm: MinimalParams) -> np.ndarray:
    """Spatial time derivative for a field of shape ``(3, ssize)``.

    Reaction terms plus ``D_i d2u_i/dx2`` from the second-order central difference on a
    periodic domain (``np.roll`` wraps the two edges).
    """
    laplacian = np.roll(u, 1, axis=1) - 2.0 * u + np.roll(u, -1, axis=1)
    return reaction(u, prm) + prm.Ds[:, None] * laplacian / prm.dx ** 2


def jacobian_sparsity(ssize: int):
    """Sparsity pattern of the Jacobian of the flattened ``(3, ssize)`` field.

    Reactions couple all three components *within* a grid cell and diffusion couples each
    component to the same cell's two periodic neighbours. Under the component-major
    flattening that is exactly a Kronecker product: a dense 3x3 block (the reactions)
    against a periodic tridiagonal in the cell index (the stencil, its corner entries
    wrapping the domain).

    Handing this to ``solve_ivp`` is what keeps the stiff solve cheap: it then estimates
    the Jacobian from a handful of grouped right-hand-side evaluations and factorises it
    sparsely, instead of building a dense ``(3 ssize)^2`` matrix.
    """
    stencil = diags([1, 1, 1, 1, 1], [-ssize + 1, -1, 0, 1, ssize - 1],
                    shape=(ssize, ssize))
    return kron(np.ones((3, 3)), stencil, format="csr")


def solve_pde(prm: MinimalParams, u0: np.ndarray, t_eval: np.ndarray,
              rtol: float = 1e-6, atol: float = 1e-9):
    """Integrate the 1D PDE from ``u0`` (shape ``(3, ssize)``) over ``t_eval`` with the
    stiff BDF solver. Returns the raw ``solve_ivp`` result, whose ``y`` is flattened."""
    u0 = np.ascontiguousarray(u0, dtype=np.float64)
    if u0.ndim != 2 or u0.shape[0] != 3:
        raise ValueError(f"u0 must have shape (3, ssize); got {u0.shape}")
    ssize = u0.shape[1]

    def f(t, y):
        return rhs_1d(y.reshape(3, ssize), prm).ravel()

    return solve_ivp(f, (t_eval[0], t_eval[-1]), u0.ravel(), method="BDF",
                     t_eval=t_eval, rtol=rtol, atol=atol,
                     jac_sparsity=jacobian_sparsity(ssize))


def log_times(t_max: float, n_times: int = 300, t_min: float = 0.1) -> np.ndarray:
    """Logarithmic recording grid spanning ``t_min`` to ``t_max``.

    Peak formation and coarsening are separated by decades, so sampling evenly per decade
    resolves both with a few hundred rows. Being a pure function of ``(t_max, n_times)``,
    it can be called independently by a run and by whatever records the time axis, and
    the two are guaranteed to agree.
    """
    return np.logspace(np.log10(min(t_min, t_max / 1e3)), np.log10(t_max), n_times)


# --------------------------------------------------------------------------- #
# Peak counting                                                               #
# --------------------------------------------------------------------------- #


def count_peaks(field: np.ndarray, min_contrast: float = 0.05,
                prominence_frac: float = 0.05) -> int:
    """Number of peaks in a periodic 1D field.

    The field is tiled three times and only maxima landing in the middle copy are
    counted, which handles the wrap-around exactly. Maxima must stand ``prominence_frac``
    of the full range above their surroundings, which discards the ripples that ride on a
    formed pattern.

    The ``min_contrast`` guard is what makes the count meaningful as a time series. Before
    the instability has grown out of the seed noise the field is essentially flat and its
    local maxima are noise: counting them naively returns one to two hundred spurious
    peaks that have nothing to do with the pattern. Requiring the relative amplitude
    ``(max - min) / mean`` to exceed a threshold returns zero until a pattern exists, so
    a count of zero is a real measurement -- the time a trace first leaves zero is the
    pattern-formation time.
    """
    field = np.asarray(field, dtype=np.float64)
    lo, hi = float(field.min()), float(field.max())
    mean = float(field.mean())
    if mean <= 0.0 or (hi - lo) / mean < min_contrast:
        return 0

    size = field.size
    peaks, _ = find_peaks(np.tile(field, 3), prominence=prominence_frac * (hi - lo))
    return int(np.count_nonzero((peaks >= size) & (peaks < 2 * size)))


def peaks_over_time(fields: np.ndarray) -> np.ndarray:
    """Apply :func:`count_peaks` to every row of a ``(n_times, ssize)`` array."""
    return np.array([count_peaks(row) for row in fields], dtype=int)


def formed_peaks(peaks: np.ndarray) -> np.ndarray:
    """Peak counts restricted to times at which a pattern exists, as floats.

    This is the reading convention for the ``peaks`` array a scan saves, and the one
    place it is written down. That array holds NaN where a replicate was never run, and
    zero where the run happened but the field was still too flat to hold peaks. Neither
    is a count of a pattern, so averaging over them would report a coarsening that did
    not happen; both become NaN here and the nan-aware statistics skip them.

    Use the array as saved, not this view, to ask *when* patterns formed: the zeros carry
    that information and are discarded here.
    """
    peaks = np.asarray(peaks, dtype=float)
    return np.where(peaks > 0, peaks, np.nan)


# --------------------------------------------------------------------------- #
# Running one system                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class PeakTrace:
    """The peak count against time from one PDE run, with the run's provenance.

    Holds no spatial profiles: the biomass and resource fields are dropped as soon as
    they have been counted, so a scan of many long runs stays small enough to keep in
    memory, to pickle back from worker processes, and to save in one npz.
    """

    t: np.ndarray            # (n_times,) recording times, logarithmically spaced
    peaks: np.ndarray        # (n_times,) peak count at each recorded time
    p: float                 # D_x / D_ext
    seed: int                # noise seed of this replicate
    k_star: float            # fastest-growing wavenumber of the fixed point
    max_growth: float        # its growth rate
    L_box: float             # domain length
    n_peaks_target: int      # wavelengths the box was sized to hold
    success: bool            # whether the solver reached t_max
    message: str             # the solver's own account of why it stopped

    @property
    def final_peaks(self) -> int:
        return int(self.peaks[-1])


def run_peaks(p: float, t_max: float, seed: int = 0, ssize: int = 2048,
              n_peaks: int = 100, n_times: int = 300, verbose: bool = True,
              **param_overrides) -> PeakTrace:
    """Evolve one system at diffusion ratio ``p`` and count its peaks over time.

    The periodic box is sized to hold ``n_peaks`` copies of the fastest-growing wavelength
    ``2 pi / k*``, so the pattern starts with a known number of peaks and any later change
    in the count is coarsening rather than an artefact of the box. The initial condition
    is the homogeneous fixed point under small multiplicative noise drawn from ``seed``;
    replicates differ only in that noise, and independence between them is the caller's
    business.

    The solution is counted and thrown away, so what comes back is the peak count against
    time and nothing more.
    """
    prm = MinimalParams.from_p(p, **param_overrides)
    u_ss = steady_state(prm)
    disp = dispersion(prm, u=u_ss)

    L_box = n_peaks * disp.wavelength
    prm = replace(prm, dx=L_box / ssize)

    if verbose:
        print(f"p={p:g}: fixed point (n, r_ext, r_x) = "
              f"({u_ss[0]:.4f}, {u_ss[1]:.4f}, {u_ss[2]:.4f})", flush=True)
        print(f"  k*={disp.k_star:.4f}, growth rate={disp.max_growth:+.5f}, "
              f"wavelength={disp.wavelength:.4f}", flush=True)
        print(f"  box: L={L_box:.3f} = {n_peaks} wavelengths, ssize={ssize}, "
              f"dx={prm.dx:.4g} ({disp.wavelength / prm.dx:.1f} points per wavelength)",
              flush=True)
        if not disp.is_turing:
            print("  warning: no Turing instability at these parameters; the pattern "
                  "will not form", flush=True)

    rng = np.random.default_rng(seed)
    u0 = u_ss[:, None] * (1.0 + NOISE_AMPLITUDE * rng.standard_normal((3, ssize)))

    sol = solve_pde(prm, u0, log_times(t_max, n_times=n_times))
    # The flattening is component-major, so the leading ssize rows are the biomass and
    # transposing them is a view; reshaping the whole of sol.y would copy all 15 MB.
    biomass = sol.y[:ssize].T

    return PeakTrace(t=sol.t, peaks=peaks_over_time(biomass), p=float(p), seed=int(seed),
                     k_star=disp.k_star, max_growth=disp.max_growth, L_box=float(L_box),
                     n_peaks_target=int(n_peaks), success=bool(sol.success),
                     message=sol.message)


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #


def p_shades(ps: Sequence[float]) -> np.ndarray:
    """Map a set of ``p`` values onto [0, 1] in log space, for colouring by mobility."""
    ps = np.asarray(ps, dtype=np.float64)
    if ps.min() >= ps.max():
        return np.full(ps.size, 0.5)
    return np.asarray(LogNorm(vmin=ps.min(), vmax=ps.max())(ps), dtype=np.float64)


def plot_peaks_vs_time(ax, t, peaks, ps, cmap=SWEEP_CMAP, n_peaks_target=None,
                       label_fmt: str = "$p={:.3g}$"):
    """Draw peak count against time: the replicate mean per ``p``, with its spread.

    ``peaks`` is ``(n_p, n_replicates, n_times)`` as saved by a scan. Replicates of one
    ``p`` differ only in their seeding noise and coarsen at slightly different moments, so
    the individual traces are shown as a shaded min-max band behind the mean rather than
    as many lines.

    Both axes are logarithmic: pattern formation and coarsening are decades apart in
    time, and the counts span more than an order of magnitude across ``p``. Each curve
    averages only over the replicates that have a pattern at that time
    (:func:`formed_peaks`), so the band shows the spread in peak count and not the spread
    in when the pattern appeared. A dashed line marks the number of peaks the boxes were
    sized to hold.
    """
    t = np.asarray(t, dtype=float)
    peaks = formed_peaks(peaks)

    for row, p, shade in zip(peaks, ps, p_shades(ps)):
        # Restrict to the times where at least one replicate has a pattern before taking
        # statistics: the nan-aware reductions warn on an all-NaN column.
        formed = np.any(np.isfinite(row), axis=0)
        if not formed.any():
            continue
        row = row[:, formed]
        color = cmap(float(shade))
        ax.fill_between(t[formed], np.nanmin(row, axis=0), np.nanmax(row, axis=0),
                        color=color, alpha=0.25, lw=0)
        ax.plot(t[formed], np.nanmean(row, axis=0), color=color, lw=2,
                label=label_fmt.format(p))

    if n_peaks_target is not None:
        ax.axhline(n_peaks_target, color=".5", ls="--", lw=1, zorder=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel("number of peaks")
    return ax


def plot_final_peaks(ax, ps, final, color=(27 / 255, 158 / 255, 119 / 255),
                     n_peaks_target=None):
    """Draw the peak count reached at ``t_max`` against ``p``, over the replicates.

    ``final`` is ``(n_p, n_replicates)``. Individual replicates are drawn as faint points
    so the scatter at a given ``p`` is visible next to their mean. Values of ``p`` where
    no pattern had formed by ``t_max`` are left out rather than plotted at zero, which a
    logarithmic count axis cannot show.
    """
    ps = np.asarray(ps, dtype=float)
    final = formed_peaks(final)

    # Same masking idiom as plot_peaks_vs_time: a p whose replicates all came back empty
    # would make the nan-aware mean warn, and is simply left off the line.
    mean = np.full(ps.size, np.nan)
    formed = np.any(np.isfinite(final), axis=1)
    mean[formed] = np.nanmean(final[formed], axis=1)

    ax.plot(np.repeat(ps, final.shape[1]), final.ravel(), "o", color=color, alpha=0.35,
            ms=5, mew=0, zorder=1)
    ax.plot(ps, mean, "o-", color=color, lw=2, ms=8, label="replicate mean", zorder=2)
    if n_peaks_target is not None:
        ax.axhline(n_peaks_target, color=".5", ls="--", lw=1, zorder=0,
                   label="linear instability ($L/\\lambda^*$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$p = D_x / D_{\\rm ext}$")
    ax.set_ylabel("number of peaks")
    return ax
