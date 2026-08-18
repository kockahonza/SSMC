"""Analytical phase-boundary functions for the Minimal Model.

Direct port of ``fr_analytics.jl`` from
``SSMC-main/src/SSMCMain/src/ModifiedMiCRM/MinimalModelSemisymbolic/``.

All functions return ``numpy.nan`` for out-of-domain inputs (the Julia
original returns ``missing``). Scalar inputs → scalar outputs; array
inputs are supported via broadcasting where noted.
"""

from __future__ import annotations

import numpy as np


def fr_ext_line_K(l: float, m: float = 1.0, c: float = 1.0, r: float = 1.0) -> float:
    """Critical K below which the community goes extinct.

    Port of ``fr_ext_line_K`` (lines 1-9).
    """
    if 0.5 <= l <= 1.0:
        return 4.0 * l * m * r / c
    if 0.0 <= l < 0.5:
        return m * r / (c * (1.0 - l))
    return float("nan")


def fr_instab_line_K(
    l: float, m: float = 1.0, c: float = 1.0, r: float = 1.0
) -> float:
    """Critical K for spatial instability at p=1 (lines 10-16)."""
    if 0.5 <= l <= 1.0:
        return m * r / (c * (1.0 - l))
    return float("nan")


def fr_cor1_instab_line_K(
    l: float, m: float = 1.0, c: float = 1.0, p: float = 1.0, r: float = 1.0
) -> float:
    """Instability boundary with diffusion ratio ``p = D_R / D_I`` (lines 17-27)."""
    if (p / (1.0 + p)) <= l <= 1.0:
        if p < (1.0 / (2.0 * (1.0 - l))):
            return l * m * r / (c * p * (1.0 - l) * (1.0 - p * (1.0 - l)))
    return float("nan")


def ksquared_to_L(k2: float, threshold: float = 0.0) -> float:
    """Convert a positive ``k²`` to a wavelength ``L = 2π / √k²`` (lines 29-36)."""
    if np.isnan(k2):
        return float("nan")
    if k2 > threshold:
        return 2.0 * np.pi / np.sqrt(k2)
    return float("nan")


# ---- v2 analytics (DI = D, DR = p*D) ----------------------------------------


def fr2_beta_lb(l: float) -> float:
    """Lower bound on β for instability (line 71)."""
    return 1.0 / (1.0 - l) if l < 0.5 else 4.0 * l


def fr2_beta_ub(l: float, p: float) -> float:
    """Upper bound on β for instability (line 72).

    Returns ``+inf`` only when the denominator is exactly zero. For negative
    denominators the formula produces a negative value (which the Julia
    original also does), and :func:`fr2_instab_beta_range` then treats the
    range as empty via the ``lb >= ub`` check.
    """
    denom = p * (1.0 - l) * (1.0 - p * (1.0 - l))
    if denom == 0.0:
        return float("inf")
    return l / denom


def fr2_instab_beta_range(
    l: float, p: float, n: int, betamax: float = 1000.0
) -> np.ndarray:
    """Return ``n`` β values spanning the instability range, or empty if none.

    Port of ``fr2_instab_beta_range`` (lines 74-85).
    """
    lb = fr2_beta_lb(l)
    ub = fr2_beta_ub(l, p)
    if lb >= ub:
        return np.array([], dtype=np.float64)
    if not np.isfinite(ub):
        ub = betamax
    return np.linspace(lb, ub, n)


def fr2_km2(
    beta: float,
    l: float,
    p: float,
    r_over_D: float,
    s: int = +1,
) -> float:
    """Squared unstable wavenumber as a function of β, l, p.

    Port of ``fr2_km2`` (lines 87-98).
    """
    chi = 1.0 - 4.0 * l / beta
    if chi < 0.0:
        return float("nan")
    rootchi = np.sqrt(chi)
    underroot = (
        (2.0 * p - 1.0) * chi
        + 2.0 * p * (1.0 - 2.0 * (1.0 - l) * p) * rootchi
        + (1.0 - 2.0 * (1.0 - l) * p) ** 2
    )
    if underroot < 0.0:
        return float("nan")
    denom = p * (1.0 - 2.0 * (1.0 - l) * p - rootchi)
    if denom == 0.0:
        return float("nan")
    return (
        r_over_D
        * ((2.0 * l) / (1.0 - rootchi))
        * ((2.0 * p * rootchi + s * np.sqrt(underroot)) / denom)
    )


# ---- Vectorised convenience wrappers ----------------------------------------

fr_ext_line_K_vec = np.vectorize(fr_ext_line_K, otypes=[float], excluded={1, 2, 3})
fr_cor1_instab_line_K_vec = np.vectorize(
    fr_cor1_instab_line_K, otypes=[float], excluded={1, 2, 3, 4}
)
fr2_km2_vec = np.vectorize(fr2_km2, otypes=[float], excluded={4})
ksquared_to_L_vec = np.vectorize(ksquared_to_L, otypes=[float], excluded={1})
