"""Dimension reduction for spatial runs: drop extinct strains.

In a sampled community most strains often collapse to zero at the homogeneous
fixed point. Because the strain equation is multiplicative
(``dN_i/dt = g_i N_i (... - m_i)``), ``N_i = 0`` is an invariant manifold: an
absent strain stays absent. Dropping such strains from the state is therefore
*exact* for the dynamics of the survivors (modulo pattern-induced local
invasion, which should be checked when a dropped strain's mean-field growth rate
could turn positive over the resource range a pattern explores).

Removing strains shrinks ``ntot`` — and hence the PDE state ``N = ntot * grid``
and the per-cell reaction cost ``O(Ns * Nr^2)`` — which speeds up every solver.
These helpers operate on parameter containers only; they do not touch the
solvers and are independent of dimensionality (intended for the 2D runs).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .params import MiCRMParams, SpatialMiCRMParams


def survivor_mask(u_ss: np.ndarray, Ns: int, thresh: float = 1e-6) -> np.ndarray:
    """Boolean mask ``(Ns,)`` of strains with steady-state biomass > ``thresh``.

    ``u_ss`` is the homogeneous fixed point, layout ``[:Ns]`` strains then
    resources.
    """
    return np.asarray(u_ss[:Ns]) > thresh


def prune_strains(
    sparams: SpatialMiCRMParams, keep: np.ndarray
) -> Tuple[SpatialMiCRMParams, np.ndarray]:
    """Return a reduced :class:`SpatialMiCRMParams` keeping only ``keep`` strains.

    ``keep`` is a boolean mask ``(Ns,)`` or an array of strain indices. Resources
    are untouched. The grid spec (``dx``, ``dy``, ``bc``) and the resource
    diffusion coefficients are preserved.

    Returns ``(reduced_sparams, keep_idx)`` where ``keep_idx`` are the surviving
    strain indices (use them to build/reconstruct an initial condition, e.g.
    :func:`prune_state`).
    """
    p = sparams.micrm
    Ns, Nr = p.Ns, p.Nr
    keep = np.asarray(keep)
    keep_idx = np.flatnonzero(keep) if keep.dtype == bool else keep.astype(int)

    reduced_micrm = MiCRMParams(
        g=p.g[keep_idx],
        w=p.w,
        m=p.m[keep_idx],
        K=p.K,
        r=p.r,
        l=p.l[keep_idx, :],
        c=p.c[keep_idx, :],
        D=p.D[keep_idx, :, :],
    )
    # Ds is (Ns + Nr,): subset the strain block, keep the resource block.
    Ds_reduced = np.concatenate([sparams.Ds[:Ns][keep_idx], sparams.Ds[Ns:]])

    reduced_sparams = SpatialMiCRMParams(
        micrm=reduced_micrm,
        Ds=Ds_reduced,
        dx=sparams.dx,
        bc=sparams.bc,
        dy=sparams.dy,
    )
    return reduced_sparams, keep_idx


def prune_state(u_full: np.ndarray, Ns: int, keep_idx: np.ndarray) -> np.ndarray:
    """Reduce a full state to surviving strains + all resources.

    Works for the homogeneous vector ``(Ns+Nr,)`` and spatial states with the
    species on axis 0 (e.g. ``(Ns+Nr, ny, nx)``): keeps strain rows ``keep_idx``
    followed by every resource row.
    """
    u_full = np.asarray(u_full)
    strains = u_full[keep_idx]
    resources = u_full[Ns:]
    return np.concatenate([strains, resources], axis=0)
