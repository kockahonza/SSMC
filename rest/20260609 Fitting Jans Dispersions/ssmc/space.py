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
