"""SSMC-python — minimal port of the Modified MiCRM Julia codebase."""

from . import analytics, linstab, minimal_model, params, plotting, rhs, sampling, solver, space

__version__ = "0.1.0"

__all__ = [
    "analytics",
    "linstab",
    "minimal_model",
    "params",
    "plotting",
    "rhs",
    "sampling",
    "solver",
    "space",
]
