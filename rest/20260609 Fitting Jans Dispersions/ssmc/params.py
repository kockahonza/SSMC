"""Parameter containers for the Modified MiCRM model.

Python port of ``BMMiCRMParams`` and ``BSMMiCRMParams`` from
``SSMC-main/src/SSMCMain/src/ModifiedMiCRM/params_basearray.jl``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BC_PERIODIC = 0
BC_CLOSED = 1
_BC_NAME_TO_CODE = {"periodic": BC_PERIODIC, "closed": BC_CLOSED}


@dataclass
class MiCRMParams:
    """Non-spatial Modified MiCRM parameters.

    All fields are ``numpy.ndarray`` with ``dtype=float64``.

    Fields
    ------
    g : (Ns,)         per-strain growth efficiency
    w : (Nr,)         per-resource energy density
    m : (Ns,)         per-strain death rate
    K : (Nr,)         per-resource external influx
    r : (Nr,)         per-resource dilution rate
    l : (Ns, Nr)      leakage fraction, strain i resource a
    c : (Ns, Nr)      consumption rate, strain i resource a
    D : (Ns, Nr, Nr)  byproduct routing: strain i routes b -> a
    """

    g: np.ndarray
    w: np.ndarray
    m: np.ndarray
    K: np.ndarray
    r: np.ndarray
    l: np.ndarray
    c: np.ndarray
    D: np.ndarray

    def __post_init__(self) -> None:
        self.g = np.ascontiguousarray(self.g, dtype=np.float64)
        self.w = np.ascontiguousarray(self.w, dtype=np.float64)
        self.m = np.ascontiguousarray(self.m, dtype=np.float64)
        self.K = np.ascontiguousarray(self.K, dtype=np.float64)
        self.r = np.ascontiguousarray(self.r, dtype=np.float64)
        self.l = np.ascontiguousarray(self.l, dtype=np.float64)
        self.c = np.ascontiguousarray(self.c, dtype=np.float64)
        self.D = np.ascontiguousarray(self.D, dtype=np.float64)

        Ns = self.g.shape[0]
        Nr = self.w.shape[0]
        expected = {
            "g": (Ns,),
            "w": (Nr,),
            "m": (Ns,),
            "K": (Nr,),
            "r": (Nr,),
            "l": (Ns, Nr),
            "c": (Ns, Nr),
            "D": (Ns, Nr, Nr),
        }
        for name, shape in expected.items():
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(
                    f"MiCRMParams.{name} has shape {arr.shape}, expected {shape}"
                )

    @property
    def Ns(self) -> int:
        return self.g.shape[0]

    @property
    def Nr(self) -> int:
        return self.w.shape[0]


@dataclass
class SpatialMiCRMParams:
    """Spatial (1D) Modified MiCRM parameters.

    Wraps a :class:`MiCRMParams` with diffusion coefficients and a grid spec.

    Fields
    ------
    micrm : MiCRMParams
    Ds    : (Ns + Nr,) diffusion coefficients (strains first, then resources)
    dx    : float       grid spacing
    bc    : {"periodic", "closed"}
    """

    micrm: MiCRMParams
    Ds: np.ndarray
    dx: float = 1.0
    bc: str = "periodic"
    bc_code: int = field(init=False)

    def __post_init__(self) -> None:
        self.Ds = np.ascontiguousarray(self.Ds, dtype=np.float64)
        n_total = self.micrm.Ns + self.micrm.Nr
        if self.Ds.shape != (n_total,):
            raise ValueError(
                f"SpatialMiCRMParams.Ds has shape {self.Ds.shape}, "
                f"expected ({n_total},) = (Ns + Nr,)"
            )
        if self.bc not in _BC_NAME_TO_CODE:
            raise ValueError(f"bc must be 'periodic' or 'closed', got {self.bc!r}")
        self.bc_code = _BC_NAME_TO_CODE[self.bc]
        self.dx = float(self.dx)

    @property
    def Ns(self) -> int:
        return self.micrm.Ns

    @property
    def Nr(self) -> int:
        return self.micrm.Nr
