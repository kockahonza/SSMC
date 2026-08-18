"""1D finite-difference diffusion stencils.

Port of the 1D branch of ``add_diffusion!`` in
``SSMC-main/src/SSMCMain/src/ModifiedMiCRM/cartesian_space.jl`` (lines 62-113).

Boundary-condition codes
------------------------
* ``0`` — periodic
* ``1`` — closed (zero-flux / Neumann)
"""

from __future__ import annotations

import numpy as np
from numba import njit

BC_PERIODIC = 0
BC_CLOSED = 1


@njit(cache=True, fastmath=False)
def add_diffusion_1d(
    du: np.ndarray,
    u: np.ndarray,
    Ds: np.ndarray,
    dx: float,
    bc_code: int,
) -> None:
    """Add ``D_i * d²u_i/dx²`` in place to ``du`` using central differences.

    ``u`` and ``du`` have shape ``(nspecies, ssize)``.
    """
    nspecies = u.shape[0]
    ssize = u.shape[1]
    dx2 = dx * dx

    # Edges
    if bc_code == BC_PERIODIC:
        for ui in range(nspecies):
            du[ui, 0] += Ds[ui] * (u[ui, ssize - 1] - 2.0 * u[ui, 0] + u[ui, 1]) / dx2
            du[ui, ssize - 1] += (
                Ds[ui]
                * (u[ui, ssize - 2] - 2.0 * u[ui, ssize - 1] + u[ui, 0])
                / dx2
            )
    elif bc_code == BC_CLOSED:
        for ui in range(nspecies):
            du[ui, 0] += Ds[ui] * (-u[ui, 0] + u[ui, 1]) / dx2
            du[ui, ssize - 1] += (
                Ds[ui] * (u[ui, ssize - 2] - u[ui, ssize - 1]) / dx2
            )
    # (Any other bc_code silently adds no flux — validated at the Python level.)

    # Bulk
    for i in range(1, ssize - 1):
        for ui in range(nspecies):
            du[ui, i] += (
                Ds[ui] * (u[ui, i - 1] - 2.0 * u[ui, i] + u[ui, i + 1]) / dx2
            )


@njit(cache=True, fastmath=False)
def add_diffusion_2d(
    du: np.ndarray,
    u: np.ndarray,
    Ds: np.ndarray,
    dx: float,
    dy: float,
    bc_code: int,
) -> None:
    """Add ``D_i * (d²u_i/dx² + d²u_i/dy²)`` in place to ``du`` (5-point stencil).

    ``u`` and ``du`` have shape ``(nspecies, ny, nx)``. Diffusion is isotropic
    per species (the same ``Ds[ui]`` on both axes), with spacings ``dx`` (x,
    last axis) and ``dy`` (y). Port of the 2D branch of ``add_diffusion!``
    (``cartesian_space.jl`` lines 115-237).

    Boundary conditions (per :data:`BC_PERIODIC` / :data:`BC_CLOSED`) are applied
    independently on each axis: periodic wraps around; closed mirrors the edge
    cell (zero-flux / Neumann, matching the 1D one-sided edge stencil).
    """
    nspecies = u.shape[0]
    ny = u.shape[1]
    nx = u.shape[2]
    dx2 = dx * dx
    dy2 = dy * dy

    for iy in range(ny):
        if bc_code == BC_PERIODIC:
            ym = ny - 1 if iy == 0 else iy - 1
            yp = 0 if iy == ny - 1 else iy + 1
        else:  # closed: out-of-domain neighbour mirrors the edge cell
            ym = iy if iy == 0 else iy - 1
            yp = iy if iy == ny - 1 else iy + 1
        for ix in range(nx):
            if bc_code == BC_PERIODIC:
                xm = nx - 1 if ix == 0 else ix - 1
                xp = 0 if ix == nx - 1 else ix + 1
            else:
                xm = ix if ix == 0 else ix - 1
                xp = ix if ix == nx - 1 else ix + 1
            for ui in range(nspecies):
                du[ui, iy, ix] += (
                    Ds[ui]
                    * (u[ui, iy, xm] - 2.0 * u[ui, iy, ix] + u[ui, iy, xp])
                    / dx2
                )
                du[ui, iy, ix] += (
                    Ds[ui]
                    * (u[ui, ym, ix] - 2.0 * u[ui, iy, ix] + u[ui, yp, ix])
                    / dy2
                )
