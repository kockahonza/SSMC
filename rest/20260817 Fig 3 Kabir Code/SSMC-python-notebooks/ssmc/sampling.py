"""Random community sampling.

Port of ``JansSampler3`` and ``get_random_metabolic_process`` from
``SSMC-main/src/SSMCMain/src/ModifiedMiCRM/RandomSystems/jans_first.jl``
(lines 346-504).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
from scipy import stats

from .params import MiCRMParams, SpatialMiCRMParams


# ---- Distribution protocol ---------------------------------------------------


class Dirac:
    """Constant "distribution" returning ``value`` on every draw."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def rvs(self, size: Any = None, random_state: Optional[np.random.Generator] = None) -> Any:
        if size is None:
            return self.value
        if isinstance(size, int):
            return np.full(size, self.value)
        return np.full(size, self.value)


def _coerce_dist(x: Any, positive: bool = False) -> Any:
    """Coerce user input into a distribution-like object with ``rvs(size, random_state=...)``.

    Accepted inputs
    ---------------
    * ``Number``              -> :class:`Dirac`
    * ``(mean, std)``         -> ``scipy.stats.norm(mean, std)`` (frozen)
    * frozen scipy.stats dist -> used as is

    Mirrors the Julia constructor's coercion logic.
    """
    if x is None:
        return None
    if isinstance(x, Dirac):
        return x
    if isinstance(x, (int, float, np.floating, np.integer)):
        return Dirac(float(x))
    if isinstance(x, tuple) and len(x) == 2:
        mean, std = x
        return stats.norm(loc=float(mean), scale=float(std))
    if hasattr(x, "rvs"):
        return x
    raise TypeError(f"cannot coerce {x!r} into a distribution")


def _sample(dist: Any, rng: np.random.Generator, size: Optional[int] = None) -> Any:
    """Uniformly handle :class:`Dirac` and scipy.stats frozen distributions."""
    if isinstance(dist, Dirac):
        return dist.rvs(size)
    return dist.rvs(size=size, random_state=rng)


# ---- Helper: random metabolic process ---------------------------------------


def _get_random_metabolic_process(
    prob_eating: float,
    c_dist: Any,
    l_dist: Any,
    num_byproducts: int,
    Nr: int,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray]:
    """Sample one (c, l, D_row) triple for strain i / resource a.

    Port of ``get_random_metabolic_process`` (lines 346-362).
    """
    if rng.random() >= prob_eating:
        return 0.0, 0.0, np.zeros(Nr)

    c = max(0.0, float(_sample(c_dist, rng)))
    l = min(1.0, max(0.0, float(_sample(l_dist, rng))))

    num_byproducts = int(min(num_byproducts, Nr))
    if num_byproducts <= 0:
        return c, l, np.zeros(Nr)

    # Sample byproduct target indices without replacement and assign uniform
    # random weights that sum to 1.
    byproducts = rng.choice(Nr, size=num_byproducts, replace=False)
    weights = rng.random(num_byproducts)
    weights /= weights.sum()
    D_row = np.zeros(Nr)
    D_row[byproducts] = weights
    return c, l, D_row


# ---- JansSampler3 ------------------------------------------------------------


@dataclass
class JansSampler3:
    """Random community sampler.

    Direct port of ``JansSampler3`` (lines 364-504). Each field below can be
    given as a scalar (-> Dirac), a ``(mean, std)`` tuple (-> Gaussian), or a
    frozen scipy.stats distribution.

    Parameters
    ----------
    Ns, Nr : int
        Number of strains and resources.
    m, r : scalar | tuple | distribution
        Strain death rate, resource dilution rate.
    num_influx_resources : int | distribution
        Number of resources that have external influx (K > 0).
    K : scalar | tuple | distribution
        Influx rate per influx resource.
    prob_eating, prob_eating_influx : float
        Probability that a given strain consumes a given (non-influx or influx)
        resource.
    num_byproducts : int | distribution
        Number of byproducts produced per consumed resource.
    c, l, cinflux, linflux :
        Consumption rate and leakage distributions for non-influx resources,
        with ``cinflux``/``linflux`` used for influx resources (default: same
        as ``c``/``l``).
    Ds, Dr, Drinflux :
        Diffusion coefficients for strains, non-influx resources, and influx
        resources.
    """

    Ns: int
    Nr: int
    m: Any = 1.0
    r: Any = 1.0
    num_influx_resources: Any = 1
    K: Any = 1.0
    prob_eating: float = 1.0
    prob_eating_influx: float = 1.0
    num_byproducts: Any = 0
    c: Any = 1.0
    l: Any = 0.5
    cinflux: Any = None
    linflux: Any = None
    Ds: Any = 1e-12
    Dr: Any = 1.0
    Drinflux: Any = None

    _coerced: bool = field(default=False, init=False, repr=False)

    def _coerce(self) -> None:
        if self._coerced:
            return
        self.m = _coerce_dist(self.m)
        self.r = _coerce_dist(self.r)
        if isinstance(self.num_influx_resources, (int, np.integer)):
            self.num_influx_resources = Dirac(int(self.num_influx_resources))
        else:
            self.num_influx_resources = _coerce_dist(self.num_influx_resources)
        self.K = _coerce_dist(self.K)
        if isinstance(self.num_byproducts, (int, np.integer)):
            self.num_byproducts = Dirac(int(self.num_byproducts))
        else:
            self.num_byproducts = _coerce_dist(self.num_byproducts)
        self.c = _coerce_dist(self.c)
        self.l = _coerce_dist(self.l)
        self.cinflux = _coerce_dist(self.cinflux) if self.cinflux is not None else self.c
        self.linflux = _coerce_dist(self.linflux) if self.linflux is not None else self.l
        self.Ds = _coerce_dist(self.Ds)
        self.Dr = _coerce_dist(self.Dr)
        self.Drinflux = _coerce_dist(self.Drinflux) if self.Drinflux is not None else self.Dr
        self._coerced = True

    def sample(
        self, rng: Optional[np.random.Generator] = None
    ) -> SpatialMiCRMParams:
        """Draw one :class:`SpatialMiCRMParams` from the sampler.

        Parameters
        ----------
        rng : numpy Generator, optional
            If ``None``, a fresh ``default_rng()`` is used.
        """
        self._coerce()
        if rng is None:
            rng = np.random.default_rng()

        Ns, Nr = self.Ns, self.Nr

        g = np.ones(Ns)
        w = np.ones(Nr)

        m = np.clip(_sample(self.m, rng, size=Ns), 0.0, np.inf)
        r = np.clip(_sample(self.r, rng, size=Nr), 0.0, np.inf)

        # Influx resources
        n_influx = int(_sample(self.num_influx_resources, rng))
        n_influx = max(0, min(n_influx, Nr))
        K = np.zeros(Nr)
        if n_influx > 0:
            influx_idx = rng.choice(Nr, size=n_influx, replace=False)
            for a in influx_idx:
                K[a] = max(0.0, float(_sample(self.K, rng)))
        else:
            influx_idx = np.array([], dtype=int)

        l_mat = np.zeros((Ns, Nr))
        c_mat = np.zeros((Ns, Nr))
        D = np.zeros((Ns, Nr, Nr))

        for a in range(Nr):
            is_influx = bool(K[a] > 0.0)
            for i in range(Ns):
                nb = int(_sample(self.num_byproducts, rng))
                if is_influx:
                    c_val, l_val, D_row = _get_random_metabolic_process(
                        self.prob_eating_influx, self.cinflux, self.linflux, nb, Nr, rng
                    )
                else:
                    c_val, l_val, D_row = _get_random_metabolic_process(
                        self.prob_eating, self.c, self.l, nb, Nr, rng
                    )
                c_mat[i, a] = c_val
                l_mat[i, a] = l_val
                D[i, :, a] = D_row

        Ds_strain = np.clip(_sample(self.Ds, rng, size=Ns), 0.0, np.inf)
        Ds_res = np.clip(_sample(self.Dr, rng, size=Nr), 0.0, np.inf)
        for a in influx_idx:
            Ds_res[a] = max(0.0, float(_sample(self.Drinflux, rng)))
        Ds = np.concatenate([np.atleast_1d(Ds_strain), np.atleast_1d(Ds_res)])

        micrm = MiCRMParams(g=g, w=w, m=m, K=K, r=r, l=l_mat, c=c_mat, D=D)
        return SpatialMiCRMParams(micrm=micrm, Ds=Ds)

    # Convenience: allow ``sampler()`` to match the Julia call syntax.
    def __call__(self, rng: Optional[np.random.Generator] = None) -> SpatialMiCRMParams:
        return self.sample(rng)
