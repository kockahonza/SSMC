"""Time-integration wrappers around :func:`scipy.integrate.solve_ivp`.

Port of the ``make_mmicrm_ss_problem`` / ``make_smmicrm_problem`` workflows,
which in the Julia code use ``DynamicSS(TRBDF2())`` and ``QNDF()`` from
``DifferentialEquations.jl``. SciPy's ``BDF`` is the closest analogue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .params import MiCRMParams, SpatialMiCRMParams
from .rhs import make_ode_rhs, make_pde_rhs, mmicrm_rhs


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
) -> PDESolution:
    """Integrate the 1D spatial PDE.

    Parameters
    ----------
    u0 : ndarray with shape ``(Ns+Nr, ssize)``
        Initial condition.
    t_span : (t0, t1)
    t_eval : optional times at which to record the solution (defaults to 100
        evenly spaced points including endpoints).
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

    sol = solve_ivp(
        f,
        t_span,
        u0.ravel(),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    u = sol.y.T.reshape(len(sol.t), ntot, ssize)
    return PDESolution(t=sol.t, u=u, success=bool(sol.success), message=sol.message)
