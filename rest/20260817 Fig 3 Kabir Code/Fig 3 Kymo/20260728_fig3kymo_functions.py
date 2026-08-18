"""Self-contained essentials for the byproduct-diffusion kymographs (Fig 3 Kymo).

Everything needed to evolve one spatially unstable single-influx community lives
here: random community sampling, well-mixed steady-state solving, linear
stability analysis, the 1D PDE, and the kymograph plotting helpers. The module
depends only on numpy, scipy, matplotlib and palettable, so the ``Consolidated``
folder runs without the ``ssmc`` package.

Where ``Fig. 3`` follows a single community, this folder compares one community
across the byproduct/influx diffusion ratio ``p = D_byprod / D_ext``. A single
community that is Turing-unstable at *every* ``p`` is drawn once
(:func:`draw_shared_community`) and re-used at each ratio with only its
diffusion coefficients rebuilt (:func:`with_byproduct_diffusion`), so the
differences between kymographs are the ratio and nothing else.

The model is the Modified MiCRM: ``Ns`` microbial strains ``N`` consume ``Nr``
resources ``R`` and excrete byproducts. In one spatial dimension,

    dN_i/dt = g_i N_i ( sum_a w_a (1 - l_ia) c_ia R_a  -  m_i )  +  Ds_i d²N_i/dx²
    dR_a/dt = K_a - r_a R_a - sum_i N_i c_ia R_a
              + sum_i sum_b D_iab (w_b / w_a) l_ib N_i c_ib R_b  +  Ds_a d²R_a/dx²

Here c is the consumption matrix, l the leakage fraction, D the byproduct
routing tensor (strain i turns resource b into resource a), K the external
influx, r the dilution rate, g / w / m the growth efficiency / energy density /
death rate, and Ds the diffusion coefficients (strains first, then resources).
The domain is periodic.

A community is "single influx" because exactly one resource is supplied
externally; the strains live off it and off each other's byproducts. When such a
community is Turing-unstable the homogeneous fixed point breaks up into a
spatial pattern, which is what the kymographs show.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

import palettable as pal

# Community size and fixed model parameters, matching the D_byprod sweep in
# scripts/20260723_task3_single_influx_pde_sweep.py.
N_STRAINS = 10           # Ns: number of strains
N_RESOURCES = 10         # Nr: number of resources
K_INFLUX = 10.0          # energy supplied to the single influx resource
L_INFLUX = 0.999         # leakage of the influx resource (almost fully leaked)
NUM_BYPRODUCTS = 2       # byproducts each consumed resource is routed into
PROB_EATING = 0.7        # chance a strain consumes a given non-influx resource
D_STRAIN = 1e-6          # strain motility (strains barely move)
D_EXT = 1.0              # influx-resource diffusion (fast); sets the scale of p

# Byproduct/influx diffusion ratios compared here: the byproducts diffuse a
# hundredth as fast as the influx resource, a tenth as fast, or equally fast.
# A Turing instability feeds on differential diffusion, so the pattern is
# fastest and finest at small p and slowest and coarsest at p = 1.
P_VALUES = (0.01, 0.1, 1.0)

# Wavenumbers scanned when reading off the dispersion relation.
KS = np.linspace(1e-3, 500.0, 2000)

# Perturbation applied to the homogeneous fixed point to seed the pattern.
NOISE_AMPLITUDE = 1e-3

# Biomass above which a strain counts as present at the homogeneous fixed point.
SURVIVOR_THRESHOLD = 1e-6

# Solver settings, named so that the generated parameter file quotes the same
# numbers the solvers actually use.
SS_T_MAX = 2e3           # time given to the well-mixed system to settle
SS_RTOL, SS_ATOL = 1e-7, 1e-9
SS_TOL = 1e-8            # max|du/dt| accepted as a steady state
PDE_RTOL, PDE_ATOL = 1e-6, 1e-9

BIOMASS_CMAP = pal.colorbrewer.sequential.YlGn_9.mpl_colormap

_EPS = np.finfo(np.float64).eps


# --------------------------------------------------------------------------- #
# Distributions                                                               #
# --------------------------------------------------------------------------- #


class Dirac:
    """Constant "distribution" that returns ``value`` on every draw."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def rvs(self, size: Any = None, random_state: Any = None) -> Any:
        if size is None:
            return self.value
        return np.full(size, self.value)


def _coerce_dist(x: Any) -> Any:
    """Turn a scalar / ``(mean, std)`` tuple / frozen distribution into one that
    exposes ``rvs``. Scalars become :class:`Dirac`, pairs become Gaussians."""
    if x is None:
        return None
    if isinstance(x, Dirac):
        return x
    if isinstance(x, (int, float, np.floating, np.integer)):
        return Dirac(float(x))
    if isinstance(x, tuple) and len(x) == 2:
        mean, std = x
        return stats.norm(loc=float(mean), scale=float(std))
    if hasattr(x, "rvs"):
        return x
    raise TypeError(f"cannot coerce {x!r} into a distribution")


def _sample(dist: Any, rng: np.random.Generator, size: Optional[int] = None) -> Any:
    """Draw from a :class:`Dirac` or a frozen scipy distribution, routing the
    generator so the whole sampler is reproducible from a single ``rng``."""
    if isinstance(dist, Dirac):
        return dist.rvs(size)
    return dist.rvs(size=size, random_state=rng)


# --------------------------------------------------------------------------- #
# Parameter containers                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class MiCRMParams:
    """Modified MiCRM reaction parameters (all arrays, float64).

    g : (Ns,)         growth efficiency
    w : (Nr,)         resource energy density
    m : (Ns,)         strain death rate
    K : (Nr,)         external influx per resource
    r : (Nr,)         resource dilution rate
    l : (Ns, Nr)      leakage fraction
    c : (Ns, Nr)      consumption rate
    D : (Ns, Nr, Nr)  byproduct routing, strain i turns resource b into a
    """

    g: np.ndarray
    w: np.ndarray
    m: np.ndarray
    K: np.ndarray
    r: np.ndarray
    l: np.ndarray
    c: np.ndarray
    D: np.ndarray

    def __post_init__(self) -> None:
        for name in ("g", "w", "m", "K", "r", "l", "c", "D"):
            setattr(self, name, np.ascontiguousarray(getattr(self, name), dtype=np.float64))
        Ns, Nr = self.g.shape[0], self.w.shape[0]
        expected = {"g": (Ns,), "w": (Nr,), "m": (Ns,), "K": (Nr,), "r": (Nr,),
                    "l": (Ns, Nr), "c": (Ns, Nr), "D": (Ns, Nr, Nr)}
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"MiCRMParams.{name} has shape "
                                 f"{getattr(self, name).shape}, expected {shape}")

    @property
    def Ns(self) -> int:
        return self.g.shape[0]

    @property
    def Nr(self) -> int:
        return self.w.shape[0]


@dataclass
class SpatialMiCRMParams:
    """A :class:`MiCRMParams` plus per-species diffusion coefficients ``Ds``
    (strains first, then resources) and the grid spacing ``dx`` of the periodic
    1D domain.

    ``dx`` is irrelevant to the well-mixed steady state and to the dispersion
    relation (which works in continuous ``k``); it is only used by
    :func:`smmicrm_rhs_1d`. The sampler therefore leaves it at its default and
    the driver rebuilds these params once the box size is known.
    """

    micrm: MiCRMParams
    Ds: np.ndarray
    dx: float = 1.0

    def __post_init__(self) -> None:
        self.Ds = np.ascontiguousarray(self.Ds, dtype=np.float64)
        n_total = self.micrm.Ns + self.micrm.Nr
        if self.Ds.shape != (n_total,):
            raise ValueError(f"Ds has shape {self.Ds.shape}, expected ({n_total},)")
        self.dx = float(self.dx)

    @property
    def Ns(self) -> int:
        return self.micrm.Ns

    @property
    def Nr(self) -> int:
        return self.micrm.Nr


# --------------------------------------------------------------------------- #
# Random community sampler                                                    #
# --------------------------------------------------------------------------- #


def _get_random_metabolic_process(
    prob_eating: float,
    c_dist: Any,
    l_dist: Any,
    num_byproducts: int,
    Nr: int,
    rng: np.random.Generator,
) -> Tuple[float, float, np.ndarray]:
    """Sample the (consumption, leakage, byproduct-routing) triple for one
    strain / resource pair.

    With probability ``1 - prob_eating`` the strain ignores the resource. Else
    it consumes at rate ``c`` with leakage ``l``, and routes leaked mass to
    ``num_byproducts`` distinct resources drawn uniformly without replacement,
    with weights ~ U(0, 1) normalised to sum to 1.
    """
    if rng.random() >= prob_eating:
        return 0.0, 0.0, np.zeros(Nr)

    c = max(0.0, float(_sample(c_dist, rng)))
    l = min(1.0, max(0.0, float(_sample(l_dist, rng))))

    num_byproducts = int(min(num_byproducts, Nr))
    if num_byproducts <= 0:
        return c, l, np.zeros(Nr)

    byproducts = rng.choice(Nr, size=num_byproducts, replace=False)
    weights = rng.random(num_byproducts)
    weights /= weights.sum()
    D_row = np.zeros(Nr)
    D_row[byproducts] = weights
    return c, l, D_row


@dataclass
class JansSampler3:
    """Random community sampler. Each strain is assigned consumption, leakage
    and byproduct routing for every resource; a subset of resources receive
    external influx. Fields accept a scalar (-> Dirac), a ``(mean, std)`` pair
    (-> Gaussian) or a frozen scipy distribution."""

    Ns: int
    Nr: int
    m: Any = 1.0
    r: Any = 1.0
    num_influx_resources: Any = 1
    K: Any = 1.0
    prob_eating: float = 1.0
    prob_eating_influx: float = 1.0
    num_byproducts: Any = 0
    c: Any = 1.0
    l: Any = 0.5
    cinflux: Any = None
    linflux: Any = None
    Ds: Any = 1e-12
    Dr: Any = 1.0
    Drinflux: Any = None

    _coerced: bool = field(default=False, init=False, repr=False)

    def _coerce(self) -> None:
        if self._coerced:
            return
        self.m = _coerce_dist(self.m)
        self.r = _coerce_dist(self.r)
        self.num_influx_resources = (
            Dirac(int(self.num_influx_resources))
            if isinstance(self.num_influx_resources, (int, np.integer))
            else _coerce_dist(self.num_influx_resources)
        )
        self.K = _coerce_dist(self.K)
        self.num_byproducts = (
            Dirac(int(self.num_byproducts))
            if isinstance(self.num_byproducts, (int, np.integer))
            else _coerce_dist(self.num_byproducts)
        )
        self.c = _coerce_dist(self.c)
        self.l = _coerce_dist(self.l)
        self.cinflux = _coerce_dist(self.cinflux) if self.cinflux is not None else self.c
        self.linflux = _coerce_dist(self.linflux) if self.linflux is not None else self.l
        self.Ds = _coerce_dist(self.Ds)
        self.Dr = _coerce_dist(self.Dr)
        self.Drinflux = _coerce_dist(self.Drinflux) if self.Drinflux is not None else self.Dr
        self._coerced = True

    def sample(self, rng: Optional[np.random.Generator] = None) -> SpatialMiCRMParams:
        """Draw one community. The order of ``rng`` draws matches the original
        Julia sampler, so a given seed reproduces a given community."""
        self._coerce()
        if rng is None:
            rng = np.random.default_rng()
        Ns, Nr = self.Ns, self.Nr

        g = np.ones(Ns)
        w = np.ones(Nr)
        m = np.clip(_sample(self.m, rng, size=Ns), 0.0, np.inf)
        r = np.clip(_sample(self.r, rng, size=Nr), 0.0, np.inf)

        # Pick which resources are externally supplied and at what rate.
        n_influx = max(0, min(int(_sample(self.num_influx_resources, rng)), Nr))
        K = np.zeros(Nr)
        if n_influx > 0:
            influx_idx = rng.choice(Nr, size=n_influx, replace=False)
            for a in influx_idx:
                K[a] = max(0.0, float(_sample(self.K, rng)))
        else:
            influx_idx = np.array([], dtype=int)

        l_mat = np.zeros((Ns, Nr))
        c_mat = np.zeros((Ns, Nr))
        D = np.zeros((Ns, Nr, Nr))
        for a in range(Nr):
            is_influx = bool(K[a] > 0.0)
            for i in range(Ns):
                nb = int(_sample(self.num_byproducts, rng))
                if is_influx:
                    c_val, l_val, D_row = _get_random_metabolic_process(
                        self.prob_eating_influx, self.cinflux, self.linflux, nb, Nr, rng)
                else:
                    c_val, l_val, D_row = _get_random_metabolic_process(
                        self.prob_eating, self.c, self.l, nb, Nr, rng)
                c_mat[i, a] = c_val
                l_mat[i, a] = l_val
                D[i, :, a] = D_row

        Ds_strain = np.clip(_sample(self.Ds, rng, size=Ns), 0.0, np.inf)
        Ds_res = np.clip(_sample(self.Dr, rng, size=Nr), 0.0, np.inf)
        for a in influx_idx:
            Ds_res[a] = max(0.0, float(_sample(self.Drinflux, rng)))
        Ds = np.concatenate([np.atleast_1d(Ds_strain), np.atleast_1d(Ds_res)])

        micrm = MiCRMParams(g=g, w=w, m=m, K=K, r=r, l=l_mat, c=c_mat, D=D)
        return SpatialMiCRMParams(micrm=micrm, Ds=Ds)

    def __call__(self, rng: Optional[np.random.Generator] = None) -> SpatialMiCRMParams:
        return self.sample(rng)


def make_sampler(
    D_byprod: float,
    K: float = K_INFLUX,
    l_influx: float = L_INFLUX,
    Ns: int = N_STRAINS,
    Nr: int = N_RESOURCES,
) -> JansSampler3:
    """Sampler for the single-influx communities of this figure.

    One resource is supplied externally at rate ``K``, eaten by every strain and
    leaked back at fraction ``l_influx``; the other resources are byproducts,
    eaten with probability ``PROB_EATING`` and leaked at a rate drawn from
    U(0.3, 0.8). Death, consumption and influx-consumption rates are lognormal
    about 1. Strains are effectively immotile (``D_STRAIN``), the influx
    resource diffuses fast (``D_EXT``), and ``D_byprod`` sets how fast the
    byproducts spread -- the parameter that controls the pattern length scale.

    The diffusion coefficients only enter through ``Ds``, so a community drawn
    at one ``D_byprod`` can be moved to another by
    :func:`with_byproduct_diffusion` rather than re-sampled.
    """
    return JansSampler3(
        Ns=Ns, Nr=Nr,
        m=stats.lognorm(s=0.3, scale=1.0),
        r=Dirac(1.0),
        num_influx_resources=1,
        K=Dirac(float(K)),
        prob_eating=PROB_EATING, prob_eating_influx=1.0,
        num_byproducts=Dirac(NUM_BYPRODUCTS),
        c=stats.lognorm(s=0.3, scale=1.0),
        l=stats.uniform(loc=0.3, scale=0.5),
        cinflux=stats.lognorm(s=0.3, scale=1.0),
        linflux=Dirac(float(l_influx)),
        Ds=Dirac(D_STRAIN),
        Dr=Dirac(float(D_byprod)),
        Drinflux=Dirac(D_EXT),
    )


def with_byproduct_diffusion(sparams: SpatialMiCRMParams, p: float,
                             dx: Optional[float] = None) -> SpatialMiCRMParams:
    """Copy a community with its byproduct diffusion set to ``p * D_EXT``.

    Only ``Ds`` (and optionally ``dx``) differs from ``sparams``; the reaction
    parameters are shared, which is what makes the comparison across ``p`` a
    comparison of one community rather than of several. ``Ds`` is rebuilt from
    this module's diffusion constants rather than edited in place, so it reads
    the same however the community was originally sampled: strains get
    ``D_STRAIN``, the externally supplied resource ``D_EXT``, and every other
    resource -- the byproducts -- the requested ratio.
    """
    Ns = sparams.Ns
    Ds = np.empty_like(sparams.Ds)
    Ds[:Ns] = D_STRAIN
    Ds[Ns:] = np.where(sparams.micrm.K > 0.0, D_EXT, p * D_EXT)
    return SpatialMiCRMParams(micrm=sparams.micrm, Ds=Ds,
                              dx=sparams.dx if dx is None else dx)


# --------------------------------------------------------------------------- #
# Dynamics                                                                    #
# --------------------------------------------------------------------------- #


def _reaction(u: np.ndarray, p: MiCRMParams) -> np.ndarray:
    """Local (space-free) Modified MiCRM terms for a field of shape
    ``(Ns + Nr, ssize)``.

    Every grid cell is evaluated at once; with ``ssize = 1`` this is the
    well-mixed right-hand side. Rows ``:Ns`` are strains ``N``, rows ``Ns:`` are
    resources ``R``.
    """
    Ns = p.Ns
    N, R = u[:Ns], u[Ns:]

    # Strains: growth from every consumed resource, minus death.
    growth = (p.w * (1.0 - p.l) * p.c) @ R                     # (Ns, ssize)
    dN = p.g[:, None] * N * (growth - p.m[:, None])

    # Resources: influx and dilution, consumption loss, byproduct production.
    uptake = R * (p.c.T @ N)                                   # sum_i N_i c_ia R_a
    leaked = np.einsum("ix,ib,bx->ibx", N, p.l * p.c, p.w[:, None] * R)
    production = np.einsum("iab,ibx->ax", p.D, leaked) / p.w[:, None]
    dR = p.K[:, None] - p.r[:, None] * R - uptake + production

    return np.concatenate([dN, dR])


def mmicrm_rhs(u: np.ndarray, p: MiCRMParams) -> np.ndarray:
    """Well-mixed time derivative for a flat state ``u`` of shape ``(Ns + Nr,)``."""
    return _reaction(u[:, None], p)[:, 0]


def smmicrm_rhs_1d(u: np.ndarray, sparams: SpatialMiCRMParams) -> np.ndarray:
    """Spatial time derivative for a field of shape ``(Ns + Nr, ssize)``.

    Reaction terms plus ``Ds_i d²u_i/dx²`` from the second-order central
    difference on a periodic domain (``np.roll`` wraps the two edges).
    """
    laplacian = (np.roll(u, 1, axis=1) - 2.0 * u + np.roll(u, -1, axis=1))
    return _reaction(u, sparams.micrm) + sparams.Ds[:, None] * laplacian / sparams.dx ** 2


def solve_steady_state(
    p: MiCRMParams,
    u0: np.ndarray,
    t_max: float = SS_T_MAX,
    rtol: float = SS_RTOL,
    atol: float = SS_ATOL,
    ss_tol: float = SS_TOL,
) -> Tuple[np.ndarray, bool]:
    """Integrate the well-mixed system to ``t_max`` with a stiff solver and report
    whether the final state is stationary (``max|du/dt| < ss_tol``).

    Returns ``(u_final, ok)``.
    """
    u0 = np.ascontiguousarray(u0, dtype=np.float64)
    sol = solve_ivp(lambda t, y: mmicrm_rhs(y, p), (0.0, t_max), u0,
                    method="BDF", rtol=rtol, atol=atol)
    if not sol.success:
        return (sol.y[:, -1] if sol.y.size else u0), False
    u_final = sol.y[:, -1]
    return u_final, bool(np.max(np.abs(mmicrm_rhs(u_final, p))) < ss_tol)


# --------------------------------------------------------------------------- #
# Linear stability                                                            #
# --------------------------------------------------------------------------- #


def make_M1(p: MiCRMParams, ss: np.ndarray) -> np.ndarray:
    """Well-mixed Jacobian of the Modified MiCRM at steady state ``ss``.

    Block structure over (strains, resources): strain-strain is diagonal
    (net growth), strain-resource and resource-strain couple the two pools, and
    resource-resource captures cross-leakage plus consumption loss and dilution.
    """
    Ns, Nr = p.Ns, p.Nr
    n = Ns + Nr
    g, w, m, r, l, c, D = p.g, p.w, p.m, p.r, p.l, p.c, p.D
    N, R = ss[:Ns], ss[Ns:]
    M1 = np.zeros((n, n))

    # Strain-strain: diagonal net growth rate.
    M1[np.arange(Ns), np.arange(Ns)] = g * ((w * (1.0 - l) * c) @ R - m)

    # Strain-resource (upper right) and resource-strain (lower left).
    M1[:Ns, Ns:] = g[:, None] * (1.0 - l) * (w * c) * N[:, None]
    reroute = np.einsum("iab,ib->ai", D, l * (w * c) * R[None, :]) / w[:, None]
    M1[Ns:, :Ns] = reroute - (c * R[None, :]).T

    # Resource-resource: cross-leakage off-diagonal, consumption + dilution on
    # the diagonal.
    M1[Ns:, Ns:] = np.einsum("iab,i->ab", D * (l * c)[:, None, :], N) \
        * (1.0 / w)[:, None] * w[None, :]
    M1[np.arange(Ns, n), np.arange(Ns, n)] += -(c * N[:, None]).sum(axis=0) - r
    return M1


def _max_real_eig(M: np.ndarray) -> float:
    return float(np.max(np.real(np.linalg.eigvals(M))))


@dataclass
class LinstabResult:
    """Dispersion relation plus a classification code (see :func:`scan_k`)."""

    ks: np.ndarray
    mrls: np.ndarray
    k0_mrl: float
    max_mrl: float
    max_k: float
    is_separated: bool
    code: int


def scan_k(
    sparams: SpatialMiCRMParams,
    ss: np.ndarray,
    ks: Sequence[float] = KS,
    zero_thr: float = 1000.0 * _EPS,
) -> LinstabResult:
    """Scan wavenumbers and classify the homogeneous steady state.

    The growth rate at wavenumber ``k`` is the largest real eigenvalue of
    ``M(k) = M1 - k**2 diag(Ds)``. The classification code encodes whether the
    well-mixed (k=0) state is stable and whether a positive peak appears at some
    ``k > 0`` (Turing instability), and whether that peak is separated from k=0
    by a stable dip.
    """
    ks = np.ascontiguousarray(ks, dtype=np.float64)
    if ks.size == 0 or ks[0] <= 0.0 or np.any(np.diff(ks) < 0):
        raise ValueError("ks must be sorted and strictly positive")

    M1 = make_M1(sparams.micrm, ss)
    k0_mrl = _max_real_eig(M1)

    # Every wavenumber at once: M1 broadcast to (len(ks), n, n) with -k² Ds on
    # the diagonal, diagonalised in one stacked call. Roughly twice as fast as
    # looping, which matters because the rejection sampler scans once per p.
    Mk = np.broadcast_to(M1, (ks.size,) + M1.shape).copy()
    diag = np.arange(M1.shape[0])
    Mk[:, diag, diag] -= (ks ** 2)[:, None] * sparams.Ds
    mrls = np.max(np.real(np.linalg.eigvals(Mk)), axis=1)

    maxi = int(np.argmax(mrls))
    max_mrl = float(mrls[maxi])
    max_positive = max_mrl > zero_thr
    is_separated = bool(np.any(mrls[:maxi] < -zero_thr))

    if k0_mrl < -zero_thr:
        code = 2 if max_positive else 1
    elif k0_mrl < zero_thr:
        code = 11 if not max_positive else (12 if is_separated else 13)
    else:
        code = 21 if not max_positive else (22 if is_separated else 23)

    return LinstabResult(ks=ks, mrls=mrls, k0_mrl=k0_mrl, max_mrl=max_mrl,
                         max_k=float(ks[maxi]), is_separated=is_separated, code=code)


def classify(result: LinstabResult) -> str:
    """Map a scan result to ``"stable"``, ``"unstable"`` (spatial/Turing) or
    ``"nospace_unstable"`` (already unstable when well mixed)."""
    if result.code in (21, 22, 23):
        return "nospace_unstable"
    if result.code in (2, 12):
        return "unstable"
    return "stable"


def count_survivors(u_ss: np.ndarray, Ns: int,
                    threshold: float = SURVIVOR_THRESHOLD) -> int:
    """Number of strains still present at the homogeneous fixed point ``u_ss``."""
    return int(np.count_nonzero(u_ss[:Ns] > threshold))


@dataclass
class SharedCommunity:
    """One community, its homogeneous fixed point, and how it behaves at each
    ``p`` -- the starting point every run in this folder shares."""

    sparams: SpatialMiCRMParams   # reaction parameters (Ds set at a reference p)
    u_ss: np.ndarray              # (Ns + Nr,) homogeneous fixed point
    dispersions: dict             # p -> LinstabResult of that ratio
    n_survivors: int              # strains alive at the fixed point
    n_draws: int                  # communities sampled before this one was kept


def draw_shared_community(
    ps: Sequence[float],
    rng: np.random.Generator,
    K: float = K_INFLUX,
    l_influx: float = L_INFLUX,
    Ns: int = N_STRAINS,
    Nr: int = N_RESOURCES,
    min_survivors: int = 0,
    max_tries: int = 200,
    ks: Sequence[float] = KS,
) -> Optional[SharedCommunity]:
    """Rejection-sample communities until one is Turing-unstable at every ``p``.

    Each draw is grown to its homogeneous fixed point from an even, low-biomass
    start and then tested three ways: it must settle without going extinct, it
    must keep at least ``min_survivors`` strains (a floor -- a richer community
    than asked for is accepted), and its fixed point must be spatially unstable
    at *all* of ``ps``. Only the last test involves diffusion, so it is applied
    by swapping ``Ds`` rather than by drawing a new community.

    Requiring instability everywhere is what makes the kymographs comparable:
    accepting a community per ``p`` instead would confound the diffusion ratio
    with community-to-community variation.

    Returns a :class:`SharedCommunity`, or ``None`` if ``max_tries`` draws all
    failed.
    """
    # The sampler needs some byproduct diffusion to hand back, but every use of
    # the community replaces it via with_byproduct_diffusion, so the first ratio
    # serves as well as any other.
    sampler = make_sampler(ps[0] * D_EXT, K=K, l_influx=l_influx, Ns=Ns, Nr=Nr)

    for attempt in range(max_tries):
        sparams = sampler(rng)
        params = sparams.micrm
        u0 = np.concatenate([0.1 * np.ones(params.Ns), params.K / np.maximum(params.r, 1e-6)])
        u_ss, ok = solve_steady_state(params, u0)
        if not ok or np.max(u_ss[:params.Ns]) < 1e-6:
            continue

        n_surv = count_survivors(u_ss, params.Ns)
        if n_surv < min_survivors:
            continue

        dispersions = {}
        for p in ps:
            result = scan_k(with_byproduct_diffusion(sparams, p), u_ss, ks)
            if classify(result) != "unstable":
                break
            dispersions[float(p)] = result
        else:
            return SharedCommunity(sparams=sparams, u_ss=u_ss, dispersions=dispersions,
                                   n_survivors=n_surv, n_draws=attempt + 1)
    return None


# --------------------------------------------------------------------------- #
# 1D PDE                                                                      #
# --------------------------------------------------------------------------- #


def jacobian_sparsity(ntot: int, ssize: int) -> csr_matrix:
    """Sparsity pattern of the Jacobian of the flattened ``(ntot, ssize)`` field.

    Reactions couple every species to every other *within* a grid cell, and
    diffusion couples each species to the same cell's two periodic neighbours --
    so the Jacobian is block-tridiagonal with dense ``ntot x ntot`` blocks.

    Handing this to ``solve_ivp`` is what makes long integrations affordable: the
    solver then estimates the Jacobian with a few dozen grouped right-hand-side
    evaluations and factorises it sparsely, instead of building and LU-decomposing
    a dense ``(ntot*ssize)²`` matrix (3.4 GB and minutes per step at ssize=1024).
    """
    index = np.arange(ntot * ssize).reshape(ntot, ssize)
    rows, cols = [], []
    for shift in (-1, 0, 1):  # this cell and its two neighbours
        neighbour = np.roll(index, shift, axis=1)
        for i in range(ntot):
            rows.append(np.repeat(index[i], ntot))
            cols.append(neighbour.T.ravel())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    n = ntot * ssize
    return csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n))


def solve_pde(
    sparams: SpatialMiCRMParams,
    u0: np.ndarray,
    t_eval: np.ndarray,
    rtol: float = PDE_RTOL,
    atol: float = PDE_ATOL,
):
    """Integrate the 1D PDE from ``u0`` (shape ``(Ns + Nr, ssize)``) over
    ``t_eval``, with the stiff BDF solver.

    Returns the raw ``solve_ivp`` result; ``sol.y`` is flattened, so use
    :func:`total_biomass` to read it back as a field.
    """
    u0 = np.ascontiguousarray(u0, dtype=np.float64)
    ntot = sparams.Ns + sparams.Nr
    if u0.ndim != 2 or u0.shape[0] != ntot:
        raise ValueError(f"u0 must have shape ({ntot}, ssize); got {u0.shape}")
    ssize = u0.shape[1]

    def f(t, y):
        return smmicrm_rhs_1d(y.reshape(ntot, ssize), sparams).ravel()

    return solve_ivp(f, (t_eval[0], t_eval[-1]), u0.ravel(), method="BDF",
                     t_eval=t_eval, rtol=rtol, atol=atol,
                     jac_sparsity=jacobian_sparsity(ntot, ssize))


def total_biomass(sol, sparams: SpatialMiCRMParams) -> np.ndarray:
    """Sum the strain rows of a flattened ``solve_ivp`` result, giving the
    ``(len(t), ssize)`` array the kymographs plot.

    The grid size is read back off the solution rather than passed in, since
    ``sol.y`` has one row per species per cell.
    """
    ntot = sparams.Ns + sparams.Nr
    ssize = sol.y.shape[0] // ntot
    return sol.y.T.reshape(len(sol.t), ntot, ssize)[:, :sparams.Ns, :].sum(axis=1)


def kymograph_times(
    t_max: float,
    n_linear: int = 400,
    n_log: int = 200,
    t_min: float = 0.1,
) -> np.ndarray:
    """Times at which to record the solution, resolving both kymographs.

    The two figures want opposite things from the recording grid: the linear-time
    plot needs rows spread evenly to ``t_max``, the log-time plot needs most of
    them packed into the early decades where the pattern actually forms. Merging
    a linear and a log grid lets a single integration serve both, at negligible
    cost (``solve_ivp`` interpolates onto ``t_eval``).
    """
    t_min = min(t_min, t_max / 1e3)
    linear = np.linspace(0.0, t_max, n_linear)
    logarithmic = np.logspace(np.log10(t_min), np.log10(t_max), n_log)
    return np.unique(np.concatenate([linear, logarithmic]))


@dataclass
class Run:
    """Everything the kymographs and their titles need from one PDE run."""

    p: float                 # byproduct/influx diffusion ratio of this run
    t: np.ndarray            # (n_times,) recording times
    x: np.ndarray            # (ssize,) grid positions
    biomass: np.ndarray      # (n_times, ssize) total strain biomass
    k_star: float            # fastest-growing wavenumber at this p
    max_mrl: float           # its growth rate
    L_box: float             # domain length
    success: bool            # whether the solver reached t_max
    message: str


def run_at_p(
    community: SharedCommunity,
    p: float,
    t_max: float,
    seed: int,
    ssize: int = 1024,
    n_peaks: float = 20.0,
    t_eval: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Run:
    """Evolve the shared community at diffusion ratio ``p`` out to ``t_max``.

    The periodic box is sized to ``n_peaks`` copies of the fastest-growing
    wavelength ``2*pi/k*`` at this ``p``, so the domain starts out able to hold
    exactly that many peaks; since ``k*`` falls as ``p`` rises, the box grows
    with ``p``. The initial condition is the homogeneous fixed point under small
    multiplicative noise: components that are extinct there stay ~0, so only the
    surviving strains and resources are effectively perturbed.
    """
    dispersion = community.dispersions[float(p)]
    k_star = dispersion.max_k
    L_box = n_peaks * 2.0 * np.pi / k_star
    dx = L_box / ssize
    sparams = with_byproduct_diffusion(community.sparams, p, dx=dx)

    # A noise stream keyed to p, so each ratio gets its own perturbation while
    # the community draw and the grid stay independent of both.
    rng_noise = np.random.default_rng(np.random.SeedSequence([seed, 1, int(round(p * 1e9))]))

    ntot = sparams.Ns + sparams.Nr
    u0 = np.tile(community.u_ss[:, None], (1, ssize)) * (
        1.0 + NOISE_AMPLITUDE * rng_noise.standard_normal((ntot, ssize)))

    if verbose:
        print(f"p={p:g}: k*={k_star:.3f}, growth rate={dispersion.max_mrl:.4g}, "
              f"L={L_box:.3f} = {n_peaks:g} wavelengths, dx={dx:.4g} "
              f"({ssize / n_peaks:.1f} points per peak)", flush=True)

    if t_eval is None:
        t_eval = kymograph_times(t_max)
    sol = solve_pde(sparams, u0, t_eval)

    return Run(p=float(p), t=sol.t, x=np.arange(ssize) * dx,
               biomass=total_biomass(sol, sparams),
               k_star=float(k_star), max_mrl=float(dispersion.max_mrl),
               L_box=float(L_box),
               success=bool(sol.success), message=sol.message)


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #


def plot_kymograph(ax, x: np.ndarray, t: np.ndarray, biomass: np.ndarray,
                   log_time: bool = False, cmap=BIOMASS_CMAP):
    """Draw total biomass against position and time.

    With ``log_time`` the time axis is logarithmic, which spreads out the early
    decades where the instability grows and the pattern first sets its
    wavelength; the ``t = 0`` row is dropped since it has no place on a log axis.
    Rows that were never integrated (NaN, as stored for an unfinished run) are
    dropped too. The mesh is rasterised so the svg stays small while axes and
    text stay vector.

    Takes the three arrays rather than a :class:`Run` so that it serves both a
    fresh run and a reload of the saved npz.
    """
    keep = ~np.all(np.isnan(biomass), axis=1)
    if log_time:
        keep &= t > 0
    t, biomass = t[keep], biomass[keep]

    im = ax.pcolormesh(x, t, biomass, cmap=cmap, shading="auto", rasterized=True)
    if log_time:
        ax.set_yscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    return im
