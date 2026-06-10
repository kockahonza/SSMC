"""Minimal Model: 1 strain, 2 resources (primary + byproduct).

Python port of ``MinimalModelV2.jl`` from
``SSMC-main/src/SSMCMain/src/ModifiedMiCRM/MinimalModelSemisymbolic/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .params import MiCRMParams, SpatialMiCRMParams


@dataclass
class MMParams:
    """Minimal Model parameter set (MinimalModelV2.MMParams).

    Parameters
    ----------
    K : float
        External influx of the primary resource.
    m : float
        Strain death rate.
    c : float
        Consumption rate of primary resource.
    l : float
        Leakage fraction of primary resource.
    d : float, optional
        Consumption rate of byproduct resource. Defaults to ``c``.
    k : float, optional
        Leakage fraction of byproduct resource (default 0).
    r : float, optional
        Resource dilution rate (default 1).
    """

    K: float
    m: float
    c: float
    l: float
    d: Optional[float] = None
    k: float = 0.0
    r: float = 1.0

    def __post_init__(self) -> None:
        if self.d is None:
            self.d = self.c


def mm_get_nospace_sol(
    mmp: MMParams,
    include_extinct: bool = False,
    threshold: Optional[float] = None,
) -> List[np.ndarray]:
    """Closed-form steady-state solutions of the Minimal Model.

    Returns up to three states ``np.array([N, I, R])``: two coexistence
    branches (if the discriminant is positive) and optionally the trivial
    extinction state.

    Port of ``mm_get_nospace_sol`` in ``MinimalModelV2.jl`` lines 48-91.
    """
    if threshold is None:
        threshold = 100.0 * np.finfo(np.float64).eps

    sols: List[np.ndarray] = []

    K, m, c, l_, d, k, r = mmp.K, mmp.m, mmp.c, mmp.l, mmp.d, mmp.k, mmp.r

    qa = c * d * m
    qb = m * r * (c + d) - K * c * d * (1.0 - l_ * k)
    qc = m * r * r - K * c * r * (1.0 - l_)

    disc = qb * qb - 4.0 * qa * qc

    def _build(N: float) -> np.ndarray:
        I_ = K / (r + c * N)
        R_ = (c * l_ * I_ * N) / (r + d * N)
        return np.array([N, I_, R_], dtype=np.float64)

    if disc > threshold:
        sqrtD = np.sqrt(disc)
        N1 = (-qb + sqrtD) / (2.0 * qa)
        if abs(N1) > threshold:
            sols.append(_build(N1))
        N2 = (-qb - sqrtD) / (2.0 * qa)
        if abs(N2) > threshold:
            sols.append(_build(N2))
    elif disc > -threshold:
        if abs(qb) > threshold:
            N = -qb / (2.0 * qa)
            sols.append(_build(N))

    if include_extinct:
        sols.append(np.array([0.0, K / r, 0.0], dtype=np.float64))

    return sols


def mmp_to_micrm(mmp: MMParams) -> MiCRMParams:
    """Convert an :class:`MMParams` to a 1-strain, 2-resource :class:`MiCRMParams`.

    Port of ``mmp_to_mmicrm`` in ``MinimalModelV2.jl`` lines 19-38.

    Ordering: index 0 is the primary (influx) resource, index 1 the byproduct.
    The D tensor routes the primary resource's byproducts to resource 1.
    """
    D = np.zeros((1, 2, 2), dtype=np.float64)
    D[0, 1, 0] = 1.0  # byproducts from resource 0 go to resource 1

    return MiCRMParams(
        g=np.array([1.0]),
        w=np.array([1.0, 1.0]),
        m=np.array([mmp.m]),
        K=np.array([mmp.K, 0.0]),
        r=np.array([mmp.r, mmp.r]),
        l=np.array([[mmp.l, mmp.k]]),
        c=np.array([[mmp.c, mmp.d]]),
        D=D,
    )


def mmp_to_smicrm(
    mmp: MMParams,
    DN: float = 1e-12,
    DI: float = 1.0,
    DR: float = 1e-12,
    dx: float = 1.0,
    bc: str = "periodic",
) -> SpatialMiCRMParams:
    """Convert an :class:`MMParams` to a :class:`SpatialMiCRMParams`.

    Diffusion coefficient ordering is (strain, influx resource, byproduct).
    Port of ``mmp_to_smmicrm`` in ``MinimalModelV2.jl`` lines 40-46.
    """
    micrm = mmp_to_micrm(mmp)
    Ds = np.array([DN, DI, DR], dtype=np.float64)
    return SpatialMiCRMParams(micrm=micrm, Ds=Ds, dx=dx, bc=bc)
