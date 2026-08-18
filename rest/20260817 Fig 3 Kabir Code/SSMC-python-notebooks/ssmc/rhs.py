"""ODE right-hand sides for the Modified MiCRM.

Python port of ``mmicrmfunc!`` and ``smmicrmfunc!`` in
``SSMC-main/src/SSMCMain/src/ModifiedMiCRM/params_basearray.jl``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numba import njit

from .params import MiCRMParams, SpatialMiCRMParams
from .space import add_diffusion_1d, add_diffusion_2d


@njit(cache=True, fastmath=False)
def _reaction_cell(
    u: np.ndarray,
    du: np.ndarray,
    g: np.ndarray,
    w: np.ndarray,
    m: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    D: np.ndarray,
) -> None:
    """Modified MiCRM reaction term for a single grid cell, written in place.

    ``u`` and ``du`` are length-``(Ns + Nr)`` 1D arrays (may be non-contiguous
    views into a larger spatial state): ``[:Ns]`` strains, ``[Ns:]`` resources.
    Single source of reaction truth for the non-spatial, 1D, and 2D RHS; port
    of ``mmicrmfunc!`` (``params_basearray.jl`` lines 51-77).
    """
    Ns = g.shape[0]
    Nr = w.shape[0]

    # Strain dynamics
    for i in range(Ns):
        s = 0.0
        for a in range(Nr):
            s += w[a] * (1.0 - l[i, a]) * c[i, a] * u[Ns + a]
        du[i] = g[i] * u[i] * (s - m[i])

    # Resource dynamics
    for a in range(Nr):
        s1 = 0.0
        for i in range(Ns):
            s1 += u[i] * c[i, a] * u[Ns + a]
        s2 = 0.0
        for i in range(Ns):
            for b in range(Nr):
                s2 += D[i, a, b] * (w[b] / w[a]) * l[i, b] * u[i] * c[i, b] * u[Ns + b]
        du[Ns + a] = K[a] - r[a] * u[Ns + a] - s1 + s2


@njit(cache=True, fastmath=False)
def mmicrm_rhs(
    u: np.ndarray,
    g: np.ndarray,
    w: np.ndarray,
    m: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Non-spatial Modified MiCRM RHS.

    State ``u`` layout: ``u[:Ns]`` strains, ``u[Ns:Ns+Nr]`` resources.
    Port of ``mmicrmfunc!`` (BMMiCRMParams, non-threaded branch), lines 51-77.
    """
    du = np.zeros(g.shape[0] + w.shape[0])
    _reaction_cell(u, du, g, w, m, K, r, l, c, D)
    return du


@njit(cache=True, fastmath=False)
def smmicrm_rhs_1d(
    u: np.ndarray,
    g: np.ndarray,
    w: np.ndarray,
    m: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    D: np.ndarray,
    Ds: np.ndarray,
    dx: float,
    bc_code: int,
) -> np.ndarray:
    """Spatial 1D Modified MiCRM RHS.

    State ``u`` has shape ``(Ns + Nr, ssize)``; returns ``du`` with the same
    shape. Port of ``smmicrmfunc!`` (lines 146-159 of ``params_basearray.jl``)
    combined with the 1D branch of ``add_diffusion!`` (lines 62-113 of
    ``cartesian_space.jl``).
    """
    Ns = g.shape[0]
    Nr = w.shape[0]
    ntot = Ns + Nr
    ssize = u.shape[1]

    du = np.zeros((ntot, ssize))

    # Reaction term at each grid cell
    for j in range(ssize):
        _reaction_cell(u[:, j], du[:, j], g, w, m, K, r, l, c, D)

    # Diffusion
    add_diffusion_1d(du, u, Ds, dx, bc_code)

    return du


@njit(cache=True, fastmath=False)
def smmicrm_rhs_2d(
    u: np.ndarray,
    g: np.ndarray,
    w: np.ndarray,
    m: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    D: np.ndarray,
    Ds: np.ndarray,
    dx: float,
    dy: float,
    bc_code: int,
) -> np.ndarray:
    """Spatial 2D Modified MiCRM RHS.

    State ``u`` has shape ``(Ns + Nr, ny, nx)``; returns ``du`` with the same
    shape. Reaction is applied per grid cell via :func:`_reaction_cell`, then a
    5-point diffusion stencil is added. Port of ``smmicrmfunc!`` over a 2D
    ``CartesianSpace`` (``params_basearray.jl`` lines 146-159 with the 2D
    branch of ``add_diffusion!``).
    """
    Ns = g.shape[0]
    Nr = w.shape[0]
    ntot = Ns + Nr
    ny = u.shape[1]
    nx = u.shape[2]

    du = np.zeros((ntot, ny, nx))

    # Reaction term at each grid cell
    for iy in range(ny):
        for ix in range(nx):
            _reaction_cell(u[:, iy, ix], du[:, iy, ix], g, w, m, K, r, l, c, D)

    # Diffusion
    add_diffusion_2d(du, u, Ds, dx, dy, bc_code)

    return du


@njit(cache=True, fastmath=False)
def smmicrm_reaction_2d(
    u: np.ndarray,
    g: np.ndarray,
    w: np.ndarray,
    m: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Reaction-only 2D RHS (no diffusion), for the IMEX-FFT stepper.

    State ``u`` has shape ``(Ns + Nr, ny, nx)``; returns ``du`` of the same
    shape containing only the local Modified MiCRM reaction term (the explicit
    part of :func:`ssmc.solver.solve_pde_2d_imex`). Diffusion is handled
    implicitly in Fourier space by the stepper, not here. Reuses the shared
    :func:`_reaction_cell` kernel; the 1D path is unaffected.
    """
    Ns = g.shape[0]
    Nr = w.shape[0]
    ntot = Ns + Nr
    ny = u.shape[1]
    nx = u.shape[2]

    du = np.zeros((ntot, ny, nx))
    for iy in range(ny):
        for ix in range(nx):
            _reaction_cell(u[:, iy, ix], du[:, iy, ix], g, w, m, K, r, l, c, D)
    return du


# -------- solve_ivp-compatible wrappers --------------------------------------


def make_ode_rhs(params: MiCRMParams) -> Callable[[float, np.ndarray], np.ndarray]:
    """Return ``f(t, y)`` callable suitable for :func:`scipy.integrate.solve_ivp`."""
    g, w, m, K, r, l, c, D = (
        params.g,
        params.w,
        params.m,
        params.K,
        params.r,
        params.l,
        params.c,
        params.D,
    )

    def f(t: float, y: np.ndarray) -> np.ndarray:
        return mmicrm_rhs(y, g, w, m, K, r, l, c, D)

    return f


def make_pde_rhs(
    sparams: SpatialMiCRMParams, ssize: int
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Return ``f(t, y)`` for a 1D spatial PDE.

    ``y`` is flattened ``(ntot * ssize,)`` in C-order: ``y.reshape(ntot, ssize)``.
    """
    p = sparams.micrm
    g, w, m, K, r, l, c, D = p.g, p.w, p.m, p.K, p.r, p.l, p.c, p.D
    Ds = sparams.Ds
    dx = sparams.dx
    bc_code = sparams.bc_code
    ntot = p.Ns + p.Nr

    def f(t: float, y: np.ndarray) -> np.ndarray:
        u = y.reshape(ntot, ssize)
        du = smmicrm_rhs_1d(u, g, w, m, K, r, l, c, D, Ds, dx, bc_code)
        return du.ravel()

    return f


def make_pde_rhs_2d(
    sparams: SpatialMiCRMParams, ny: int, nx: int
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Return ``f(t, y)`` for a 2D spatial PDE.

    ``y`` is flattened ``(ntot * ny * nx,)`` in C-order:
    ``y.reshape(ntot, ny, nx)``.
    """
    p = sparams.micrm
    g, w, m, K, r, l, c, D = p.g, p.w, p.m, p.K, p.r, p.l, p.c, p.D
    Ds = sparams.Ds
    dx = sparams.dx
    dy = sparams.dy
    bc_code = sparams.bc_code
    ntot = p.Ns + p.Nr

    def f(t: float, y: np.ndarray) -> np.ndarray:
        u = y.reshape(ntot, ny, nx)
        du = smmicrm_rhs_2d(u, g, w, m, K, r, l, c, D, Ds, dx, dy, bc_code)
        return du.ravel()

    return f
