"""Matplotlib plotting helpers.

Mirrors the conventions of ``plotting.jl``, ``plotting_util.jl``, and
``figures_util.jl`` in the Julia codebase.

Colour scheme (matches Julia ``PaperColors``):

* extinct — blue
* stable — orange
* Turing unstable — green
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import analytics
from .params import MiCRMParams

STATE_COLORS: Dict[str, str] = {
    "extinct": "#1f77b4",          # blue
    "stable": "#d95f02",           # orange
    "unstable": "#1b9e77",         # green
    "nospace_unstable": "#7570b3", # purple (rare; not a forfigures state)
}


def setup_mm_Kl_ax(ax, K_lim=(1e-1, 1e3), l_lim=(0.0, 1.0)) -> None:
    """Configure a phase-diagram axis (log K vs. leakage l)."""
    ax.set_xscale("log")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$l$")
    ax.set_xlim(*K_lim)
    ax.set_ylim(*l_lim)


def draw_fr_ext_line(
    ax,
    m: float = 1.0,
    c: float = 1.0,
    r: float = 1.0,
    ls: Optional[np.ndarray] = None,
    **plot_kwargs,
):
    """Overlay the extinction boundary (solid black by default)."""
    if ls is None:
        ls = np.linspace(1e-3, 1 - 1e-3, 400)
    Ks = np.array([analytics.fr_ext_line_K(float(l), m=m, c=c, r=r) for l in ls])
    mask = np.isfinite(Ks)
    style = {"color": "k", "linestyle": "-", "linewidth": 1.5}
    style.update(plot_kwargs)
    return ax.plot(Ks[mask], ls[mask], **style)


def draw_fr_cor1_instab_line(
    ax,
    p: float,
    m: float = 1.0,
    c: float = 1.0,
    r: float = 1.0,
    ls: Optional[np.ndarray] = None,
    **plot_kwargs,
):
    """Overlay the instability boundary for diffusion ratio ``p`` (dashed by default)."""
    if ls is None:
        ls = np.linspace(1e-3, 1 - 1e-3, 400)
    Ks = np.array(
        [analytics.fr_cor1_instab_line_K(float(l), m=m, c=c, p=p, r=r) for l in ls]
    )
    mask = np.isfinite(Ks)
    style = {"color": "k", "linestyle": "--", "linewidth": 1.2}
    style.update(plot_kwargs)
    return ax.plot(Ks[mask], ls[mask], **style)


def plot_dispersion(ax, ks: np.ndarray, mrls: np.ndarray, imags: Optional[np.ndarray] = None):
    """Plot dispersion relation λ_max(k).

    If ``imags`` is provided, imaginary parts are drawn dashed. Includes a
    zero line for quick stability reading.
    """
    ax.plot(ks, mrls, color="C0", label=r"Re $\lambda_{\max}$")
    if imags is not None:
        ax.plot(ks, imags, color="C0", linestyle="--", label=r"Im $\lambda_{\max}$")
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\lambda_{\max}$")


def plot_1d_snapshot(ax, u: np.ndarray, dx: float, params: MiCRMParams):
    """Plot a 1D spatial snapshot with strains in viridis and resources in plasma.

    ``u`` has shape ``(Ns+Nr, ssize)``.
    """
    import matplotlib.pyplot as plt

    Ns, Nr = params.Ns, params.Nr
    ssize = u.shape[1]
    x = np.arange(ssize) * dx

    strain_cmap = plt.get_cmap("viridis")
    resource_cmap = plt.get_cmap("plasma")

    for i in range(Ns):
        ax.plot(
            x,
            u[i],
            color=strain_cmap(i / max(Ns - 1, 1)),
            label=f"N{i + 1}",
        )
    for a in range(Nr):
        ax.plot(
            x,
            u[Ns + a],
            color=resource_cmap(a / max(Nr - 1, 1)),
            linestyle="--",
            label=f"R{a + 1}",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("concentration")


def plot_2d_field(ax, field2d: np.ndarray, dx: float, dy: Optional[float] = None,
                  cmap: str = "viridis", **pcm_kwargs):
    """Plot a single species' 2D field ``(ny, nx)`` as a heatmap.

    Uses ``pcolormesh`` with physical extent set by ``dx`` (x) and ``dy`` (y;
    defaults to ``dx``). Returns the mesh handle for colorbar attachment.
    """
    if dy is None:
        dy = dx
    ny, nx = field2d.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    mesh = ax.pcolormesh(x, y, field2d, cmap=cmap, shading="auto", **pcm_kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return mesh


def plot_2d_snapshot(u: np.ndarray, dx: float, params: MiCRMParams,
                     dy: Optional[float] = None, which: str = "strains",
                     ncols: int = 5, panel_size: float = 2.6):
    """Grid of per-species 2D field panels (2D analogue of ``plot_1d_snapshot``).

    ``u`` has shape ``(Ns+Nr, ny, nx)``. ``which`` selects ``"strains"`` (the
    first ``Ns`` fields, viridis) or ``"resources"`` (the last ``Nr`` fields,
    plasma). Returns ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt

    Ns, Nr = params.Ns, params.Nr
    if which == "strains":
        idx, cmap, prefix = range(Ns), "viridis", "N"
    elif which == "resources":
        idx, cmap, prefix = range(Ns, Ns + Nr), "plasma", "R"
    else:
        raise ValueError("which must be 'strains' or 'resources'")
    idx = list(idx)

    n = len(idx)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows),
        squeeze=False, constrained_layout=True,
    )
    flat = axes.ravel()
    for panel, k in enumerate(idx):
        ax = flat[panel]
        mesh = plot_2d_field(ax, u[k], dx, dy, cmap=cmap)
        ax.set_title(f"{prefix}{panel + 1}")
        fig.colorbar(mesh, ax=ax, shrink=0.85)
    for ax in flat[n:]:
        ax.axis("off")
    return fig, axes
