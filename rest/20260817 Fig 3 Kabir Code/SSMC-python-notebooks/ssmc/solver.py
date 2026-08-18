"""Time-integration wrappers around :func:`scipy.integrate.solve_ivp`.

Port of the ``make_mmicrm_ss_problem`` / ``make_smmicrm_problem`` workflows,
which in the Julia code use ``DynamicSS(TRBDF2())`` and ``QNDF()`` from
``DifferentialEquations.jl``. SciPy's ``BDF`` is the closest analogue.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from .params import MiCRMParams, SpatialMiCRMParams
from .rhs import (
    make_ode_rhs,
    make_pde_rhs,
    make_pde_rhs_2d,
    mmicrm_rhs,
    smmicrm_reaction_2d,
)


@lru_cache(maxsize=None)
def pde_jac_sparsity(ntot: int, ssize: int) -> sp.csr_matrix:
    """Sparsity pattern of the 1D PDE Jacobian for ``solve_ivp``.

    Flattened state uses C-order ``index(i, j) = i * ssize + j`` (species ``i``,
    grid cell ``j``), matching :func:`make_pde_rhs`. Two coupling sources:

    * reaction — within each grid cell every species couples to every other,
      giving a dense ``ntot x ntot`` block (conservative; the true block is
      slightly sparser);
    * diffusion — each ``(i, j)`` couples to its spatial neighbours
      ``(i, j-1)`` and ``(i, j+1)`` (periodic wrap).

    Supplying this to BDF replaces the dense ``N x N`` finite-difference
    Jacobian (here ``N = ntot * ssize``) with a sparse one and lets SciPy
    group columns by graph coloring, cutting both RAM and RHS evaluations per
    Jacobian rebuild by orders of magnitude. Cached per ``(ntot, ssize)``.
    """
    rows, cols = [], []
    # Reaction: full ntot x ntot block within each grid cell.
    species = np.arange(ntot)
    for j in range(ssize):
        idx = species * ssize + j
        I, J = np.meshgrid(idx, idx, indexing="ij")
        rows.append(I.ravel())
        cols.append(J.ravel())
    # Diffusion: each (i, j) couples to (i, j +/- 1), periodic.
    i = species[:, None]
    j = np.arange(ssize)[None, :]
    center = (i * ssize + j).ravel()
    rows += [center, center]
    cols += [
        (i * ssize + (j - 1) % ssize).ravel(),
        (i * ssize + (j + 1) % ssize).ravel(),
    ]
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    N = ntot * ssize
    return sp.csr_matrix((np.ones(r.size, dtype=np.int8), (r, c)), shape=(N, N))


@lru_cache(maxsize=None)
def pde_jac_sparsity_2d(Ns: int, Nr: int, ny: int, nx: int) -> sp.csr_matrix:
    """Sparsity pattern of the 2D PDE Jacobian for ``solve_ivp``.

    Flattened state uses C-order ``index(i, iy, ix) = i*(ny*nx) + iy*nx + ix``
    with ``ntot = Ns + Nr``, matching :func:`make_pde_rhs_2d`. Built as a
    Kronecker sum so it stays cheap to construct even for large grids:

    * reaction — ``kron(A, I_ncell)`` where ``A`` is the exact per-cell species
      coupling: a strain row depends on itself and all resources; a resource row
      depends on every species (see :func:`ssmc.rhs._reaction_cell`);
    * diffusion — ``kron(I_ntot, L)`` where ``L`` is the 5-point grid adjacency
      (self + 4 neighbours). Neighbour wrap is always included (a structural
      superset that is valid for both periodic and closed BCs).

    A dense ``N x N`` Jacobian is infeasible in 2D (``N = ntot*ny*nx``); this
    pattern lets BDF use a sparse Jacobian with ~O(ntot) finite-difference
    colors regardless of grid size. Cached per ``(Ns, Nr, ny, nx)``.
    """
    ntot = Ns + Nr
    # Exact per-cell species coupling block A (ntot x ntot).
    A = np.zeros((ntot, ntot), dtype=np.int8)
    A[:Ns, :Ns] = np.eye(Ns, dtype=np.int8)  # strain depends on itself only
    A[:Ns, Ns:] = 1                           # ... and on all resources
    A[Ns:, :] = 1                             # resource depends on every species

    def _cycle_adj(n: int) -> sp.csr_matrix:
        if n == 1:
            return sp.csr_matrix(np.ones((1, 1)))
        diags = [np.ones(n - 1), np.ones(n), np.ones(n - 1)]
        T = sp.diags(diags, [-1, 0, 1], format="lil")
        T[0, n - 1] = 1.0
        T[n - 1, 0] = 1.0
        return T.tocsr()

    ncell = ny * nx
    L = sp.kron(_cycle_adj(ny), sp.identity(nx)) + sp.kron(
        sp.identity(ny), _cycle_adj(nx)
    )
    S = sp.kron(sp.csr_matrix(A), sp.identity(ncell)) + sp.kron(
        sp.identity(ntot), L
    )
    return (S != 0).astype(np.int8).tocsr()


@dataclass
class PDESolution:
    """Lightweight container returned by :func:`solve_pde`."""

    t: np.ndarray
    u: np.ndarray  # shape (len(t), ntot, ssize)
    success: bool
    message: str


def solve_steady_state(
    params: MiCRMParams,
    u0: np.ndarray,
    t_max: float = 1e5,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    ss_tol: float = 1e-8,
    n_checks: int = 20,
) -> Tuple[np.ndarray, bool]:
    """Integrate until ``||du/dt||∞ < ss_tol``, or until ``t_max``.

    Returns ``(u_ss, converged)``. Uses ``solve_ivp(method="BDF")``.
    """
    u0 = np.ascontiguousarray(u0, dtype=np.float64)
    f = make_ode_rhs(params)

    # Checkpoints; at each we look at du
    t_eval = np.linspace(0.0, t_max, n_checks + 1)[1:]

    sol = solve_ivp(
        f,
        (0.0, t_max),
        u0,
        method="BDF",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    if not sol.success:
        try:
            return sol.y[:, -1], False
        except Exception:
            return u0, False

    # Walk back from the final time checking ||du||
    u_final = sol.y[:, -1]
    du = mmicrm_rhs(
        u_final, params.g, params.w, params.m, params.K, params.r, params.l, params.c, params.D
    )
    converged = bool(np.max(np.abs(du)) < ss_tol)
    return u_final, converged


def solve_pde(
    sparams: SpatialMiCRMParams,
    u0: np.ndarray,
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: str = "BDF",
    use_jac_sparsity: bool = True,
) -> PDESolution:
    """Integrate the 1D spatial PDE.

    Parameters
    ----------
    u0 : ndarray with shape ``(Ns+Nr, ssize)``
        Initial condition.
    t_span : (t0, t1)
    t_eval : optional times at which to record the solution (defaults to 100
        evenly spaced points including endpoints).
    use_jac_sparsity : pass the analytic sparsity pattern (see
        :func:`pde_jac_sparsity`) to the implicit solver. Default ``True``;
        only the implicit methods (``BDF``, ``Radau``, ``LSODA``) use it.
    """
    u0 = np.ascontiguousarray(u0, dtype=np.float64)
    ntot = sparams.Ns + sparams.Nr
    if u0.ndim != 2 or u0.shape[0] != ntot:
        raise ValueError(
            f"u0 must have shape ({ntot}, ssize); got {u0.shape}"
        )
    ssize = u0.shape[1]

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 101)

    f = make_pde_rhs(sparams, ssize)

    jac_sparsity = pde_jac_sparsity(ntot, ssize) if use_jac_sparsity else None

    sol = solve_ivp(
        f,
        t_span,
        u0.ravel(),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        jac_sparsity=jac_sparsity,
        dense_output=False,
    )

    u = sol.y.T.reshape(len(sol.t), ntot, ssize)
    return PDESolution(t=sol.t, u=u, success=bool(sol.success), message=sol.message)


def solve_pde_2d(
    sparams: SpatialMiCRMParams,
    u0: np.ndarray,
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: str = "BDF",
    use_jac_sparsity: bool = True,
) -> PDESolution:
    """Integrate the 2D spatial PDE.

    Parameters
    ----------
    u0 : ndarray with shape ``(Ns+Nr, ny, nx)``
        Initial condition (species, row y, col x).
    t_span : (t0, t1)
    t_eval : optional times at which to record the solution (defaults to 101
        evenly spaced points including endpoints).
    use_jac_sparsity : pass the analytic sparsity pattern (see
        :func:`pde_jac_sparsity_2d`) to the implicit solver. Default ``True`` —
        in 2D a dense Jacobian is infeasible, so leave this on.

    Returns
    -------
    PDESolution with ``u`` of shape ``(len(t), Ns+Nr, ny, nx)``.
    """
    u0 = np.ascontiguousarray(u0, dtype=np.float64)
    ntot = sparams.Ns + sparams.Nr
    if u0.ndim != 3 or u0.shape[0] != ntot:
        raise ValueError(
            f"u0 must have shape ({ntot}, ny, nx); got {u0.shape}"
        )
    ny, nx = u0.shape[1], u0.shape[2]

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 101)

    f = make_pde_rhs_2d(sparams, ny, nx)

    jac_sparsity = (
        pde_jac_sparsity_2d(sparams.Ns, sparams.Nr, ny, nx)
        if use_jac_sparsity
        else None
    )

    sol = solve_ivp(
        f,
        t_span,
        u0.ravel(),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        jac_sparsity=jac_sparsity,
        dense_output=False,
    )

    u = sol.y.T.reshape(len(sol.t), ntot, ny, nx)
    return PDESolution(t=sol.t, u=u, success=bool(sol.success), message=sol.message)


def diffusion_symbol_2d(
    Ds: np.ndarray, ny: int, nx: int, dx: float, dy: float
) -> np.ndarray:
    """Per-species symbol of the FD diffusion operator under ``rfft2``.

    Returns an array of shape ``(ntot, ny, nx//2 + 1)`` giving
    ``ell[i, ky, kx] = Ds[i] * Lsym[ky, kx]`` (<= 0), where ``Lsym`` is the
    eigenvalue of the periodic central-difference Laplacian:
    ``Lsym = -(2 - 2cos(2*pi*fy))/dy^2 - (2 - 2cos(2*pi*fx))/dx^2`` with
    ``fy = fftfreq(ny)``, ``fx = rfftfreq(nx)``.

    Using the *discrete* symbol (not the spectral ``-k^2``) makes the IMEX
    stepper treat diffusion identically to :func:`ssmc.space.add_diffusion_2d`,
    so the discretization and dispersion relation match the FD/BDF path.
    """
    fy = np.fft.fftfreq(ny)            # (ny,)
    fx = np.fft.rfftfreq(nx)           # (nx//2 + 1,)
    lam_y = -(2.0 - 2.0 * np.cos(2.0 * np.pi * fy)) / (dy * dy)
    lam_x = -(2.0 - 2.0 * np.cos(2.0 * np.pi * fx)) / (dx * dx)
    Lsym = lam_y[:, None] + lam_x[None, :]      # (ny, nxf)
    return Ds[:, None, None] * Lsym[None, :, :]  # (ntot, ny, nxf)


def solve_pde_2d_imex(
    sparams: SpatialMiCRMParams,
    u0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    t_eval: Optional[np.ndarray] = None,
    order: int = 2,
) -> PDESolution:
    """Integrate the 2D PDE with a semi-implicit IMEX-FFT stepper (SBDF2).

    Diffusion (linear, periodic, constant per-species ``Ds``) is treated
    implicitly in Fourier space — diagonal, so each implicit solve is a forward
    + inverse FFT with no matrix factorization or fill-in (``O(N log N)`` per
    step). The Modified MiCRM reaction is treated explicitly via
    :func:`ssmc.rhs.smmicrm_reaction_2d`. The implicit (diffusion) part is
    unconditionally stable, so ``dt`` is limited only by the grid-independent
    reaction kinetics — the scalable alternative to :func:`solve_pde_2d` for
    long runs on fine grids.

    Parameters
    ----------
    u0 : ndarray with shape ``(Ns+Nr, ny, nx)``.
    t_span : (t0, t1).
    dt : fixed time step.
    t_eval : times to record (defaults to 101 evenly spaced incl. endpoints).
        Each requested time is recorded at the first step that reaches it; the
        actual step times are returned in ``PDESolution.t``.
    order : 2 for SBDF2 (default), 1 for IMEX-Euler.

    Requires ``sparams.bc == "periodic"`` (FFT). Returns a ``PDESolution`` with
    ``u`` of shape ``(len(t), Ns+Nr, ny, nx)``.
    """
    if sparams.bc != "periodic":
        raise ValueError(
            "solve_pde_2d_imex requires periodic boundary conditions "
            f"(FFT-based); got bc={sparams.bc!r}. Use solve_pde_2d for closed BC."
        )
    u0 = np.ascontiguousarray(u0, dtype=np.float64)
    ntot = sparams.Ns + sparams.Nr
    if u0.ndim != 3 or u0.shape[0] != ntot:
        raise ValueError(f"u0 must have shape ({ntot}, ny, nx); got {u0.shape}")
    ny, nx = u0.shape[1], u0.shape[2]
    t0, t1 = t_span
    if t_eval is None:
        t_eval = np.linspace(t0, t1, 101)
    t_eval = np.asarray(t_eval, dtype=np.float64)

    p = sparams.micrm
    g, w, m, K, r, l, c, D = p.g, p.w, p.m, p.K, p.r, p.l, p.c, p.D

    def react(u):
        return smmicrm_reaction_2d(u, g, w, m, K, r, l, c, D)

    ell = diffusion_symbol_2d(sparams.Ds, ny, nx, sparams.dx, sparams.dy)
    denom1 = 1.0 / dt - ell          # IMEX-Euler / SBDF1 denominator
    denom2 = 1.5 / dt - ell          # SBDF2 denominator

    nsteps = int(np.ceil((t1 - t0) / dt))

    # Output buffers; record the first step that reaches each requested time.
    rec_t, rec_u = [], []
    next_eval = 0

    def _maybe_record(t_now, u_now):
        nonlocal next_eval
        while next_eval < t_eval.size and t_now >= t_eval[next_eval] - 1e-9:
            rec_t.append(t_now)
            rec_u.append(u_now.copy())
            next_eval += 1

    u_curr = u0
    _maybe_record(t0, u_curr)

    Nl_curr = react(u_curr)
    u_prev = None
    Nl_prev = None
    for step in range(nsteps):
        if step == 0 or order == 1:
            rhs = u_curr / dt + Nl_curr
            u_next = np.fft.irfft2(
                np.fft.rfft2(rhs, axes=(-2, -1)) / denom1, s=(ny, nx), axes=(-2, -1)
            )
        else:
            rhs = (2.0 * u_curr - 0.5 * u_prev) / dt + (2.0 * Nl_curr - Nl_prev)
            u_next = np.fft.irfft2(
                np.fft.rfft2(rhs, axes=(-2, -1)) / denom2, s=(ny, nx), axes=(-2, -1)
            )
        t = t0 + (step + 1) * dt
        u_prev, Nl_prev = u_curr, Nl_curr
        u_curr = u_next
        Nl_curr = react(u_curr)
        _maybe_record(t, u_curr)

    success = bool(np.all(np.isfinite(u_curr)))
    msg = "completed" if success else "non-finite state encountered"
    return PDESolution(
        t=np.asarray(rec_t),
        u=np.asarray(rec_u),
        success=success,
        message=msg,
    )
