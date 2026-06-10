"""Linear stability analysis for the Modified MiCRM.

Port of ``SSMC-main/src/SSMCMain/src/ModifiedMiCRM/linstab/linstab.jl``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .params import MiCRMParams, SpatialMiCRMParams

_EPS = np.finfo(np.float64).eps


def make_M1(params: MiCRMParams, ss: np.ndarray) -> np.ndarray:
    """Construct the non-spatial Jacobian M1 at steady state ``ss``.

    Returns an ``(Ns+Nr, Ns+Nr)`` dense matrix. Direct port of ``make_M1!``
    (lines 4-46).

    Block structure
    ---------------
    * Strain-strain (T): diagonal only.
    * Strain-resource (U): upper-right block.
    * Resource-strain (V): lower-left block, accounts for leakage rerouting
      and direct consumption.
    * Resource-resource (W): off-diagonal = cross-leakage, diagonal = direct
      consumption loss + dilution.
    """
    Ns, Nr = params.Ns, params.Nr
    N = Ns + Nr
    g, w, m, r = params.g, params.w, params.m, params.r
    l, c, D = params.l, params.c, params.D
    M1 = np.zeros((N, N), dtype=np.float64)

    # T: strain-strain (diagonal only)
    for i in range(Ns):
        acc = 0.0
        for a in range(Nr):
            acc += g[i] * (1.0 - l[i, a]) * w[a] * c[i, a] * ss[Ns + a]
        acc -= g[i] * m[i]
        M1[i, i] = acc

    # U and V
    for i in range(Ns):
        for a in range(Nr):
            # U: strain-resource
            M1[i, Ns + a] = g[i] * (1.0 - l[i, a]) * w[a] * c[i, a] * ss[i]
            # V: resource-strain
            v = 0.0
            for b in range(Nr):
                v += D[i, a, b] * l[i, b] * (w[b] / w[a]) * c[i, b] * ss[Ns + b]
            v -= c[i, a] * ss[Ns + a]
            M1[Ns + a, i] = v

    # W: resource-resource
    for a in range(Nr):
        for b in range(Nr):
            acc = 0.0
            for i in range(Ns):
                acc += D[i, a, b] * l[i, b] * (w[b] / w[a]) * c[i, b] * ss[i]
            M1[Ns + a, Ns + b] = acc
        diag = 0.0
        for i in range(Ns):
            diag -= c[i, a] * ss[i]
        diag -= r[a]
        M1[Ns + a, Ns + a] += diag

    return M1


def M1_to_M(M1: np.ndarray, Ds: np.ndarray, k: float) -> np.ndarray:
    """Return ``M(k) = M1 - k² diag(Ds)`` (does not modify ``M1``)."""
    M = M1.copy()
    k2 = k * k
    for i in range(len(Ds)):
        M[i, i] -= k2 * Ds[i]
    return M


@dataclass
class LinstabResult:
    """Result of :func:`scan_k`."""

    ks: np.ndarray
    mrls: np.ndarray  # max Re(eigenvalue) at each k > 0
    k0_mrl: float  # max Re(eigenvalue) at k = 0
    max_mrl: float
    max_k: float
    is_separated: bool  # True if a k<max has MRL < -zero_thr (dip between k=0 and peak)
    code: int  # 1/2/11/12/13/21/22/23 per LinstabScanTester2


def scan_k(
    sparams: SpatialMiCRMParams,
    ss: np.ndarray,
    ks: Sequence[float],
    zero_thr: float = 1000.0 * _EPS,
    peak_thr: Optional[float] = None,
) -> LinstabResult:
    """Scan wavenumbers ``ks`` and classify stability.

    Direct port of ``LinstabScanTester2`` (lines 157-242). Returns the
    dispersion array plus a diagnostic code:

    * ``1``  — stable everywhere (k=0 stable, no positive peak)
    * ``2``  — Turing unstable (k=0 stable, positive peak at some k>0)
    * ``11`` — k0 near zero, no positive peak (marginal / degenerate)
    * ``12`` — k0 near zero, separated positive peak
    * ``13`` — k0 near zero, peak connected to k=0 (messy)
    * ``21/22/23`` — k0 is positive: analogous sub-cases (non-spatial instability)
    """
    if peak_thr is None:
        peak_thr = zero_thr

    ks = np.ascontiguousarray(ks, dtype=np.float64)
    if ks.size == 0 or ks[0] <= 0.0 or np.any(np.diff(ks) < 0):
        raise ValueError("ks must be a sorted sequence of strictly positive floats")

    M1 = make_M1(sparams.micrm, ss)

    # k = 0 case
    k0_mrl = float(np.max(np.real(np.linalg.eigvals(M1))))

    # dispersion relation
    Ds = sparams.Ds
    mrls = np.empty(ks.size, dtype=np.float64)
    for ki, k in enumerate(ks):
        Mk = M1_to_M(M1, Ds, float(k))
        mrls[ki] = float(np.max(np.real(np.linalg.eigvals(Mk))))

    maxi = int(np.argmax(mrls))
    max_mrl = float(mrls[maxi])
    max_k = float(ks[maxi])

    max_positive = max_mrl > peak_thr

    # "separated" = there is a dip (MRL < -zero_thr) somewhere between k=0 and the peak
    is_separated = False
    for j in range(maxi):
        if mrls[j] < -zero_thr:
            is_separated = True
            break

    # classification
    if k0_mrl < -zero_thr:
        code = 2 if max_positive else 1
    elif k0_mrl < zero_thr:
        if not max_positive:
            code = 11
        elif is_separated:
            code = 12
        else:
            code = 13
    else:
        if not max_positive:
            code = 21
        elif is_separated:
            code = 22
        else:
            code = 23

    return LinstabResult(
        ks=ks,
        mrls=mrls,
        k0_mrl=k0_mrl,
        max_mrl=max_mrl,
        max_k=max_k,
        is_separated=is_separated,
        code=code,
    )


def classify(result: LinstabResult) -> str:
    """Map a :class:`LinstabResult` to ``"stable"``, ``"unstable"``, or
    ``"nospace_unstable"``.

    Mirrors ``get_simplified_analysis`` in ``MinimalModelV2.jl`` lines 217-230
    for a single steady state.
    """
    if result.code in (21, 22, 23):
        return "nospace_unstable"
    if result.code in (2, 12):
        return "unstable"
    return "stable"
