#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_micrm_params.py

Dump the MiCRM parameters and steady states from the `data/comm_*.dat` pickles
of Marsland III, Cui & Mehta, Sci. Rep. 10:3308 (2020) into one HDF5 file per
input, under `micrm_params_used/`.

    python extract_micrm_params.py

No options and no filtering: everything is written as-is, and any thresholding
(who counts as extinct, which wells to use) is left to the analysis side.

MODEL
-----
Every dataset uses the same equations; g, w, l and tau are scalars throughout:

    dN_i/dt = g N_i [ sum_a (1-l) w c_ia R_a  -  m_i ]

    dR_a/dt = (R0_a - R_a)/tau - sum_i N_i c_ia R_a
              + sum_b D_ab l R_b ( sum_i N_i c_ib )

CONTENTS
--------
    c (S_tot, M)   D (M, M)   m (S_tot,)   m_offset (n_wells,)
    R0 (M, n_wells)           N, R = steady state
    g, w, l, tau              root attributes
    wells/name, wells/clean, and the dataset's own per-well metadata
    labels/species_family, labels/resource_type   (where the source has them)

The maintenance vector for well k is `m + m_offset[k]`; the EMP sweeps add one
scalar per well, recoverable exactly from the stored metadata, and elsewhere
m_offset is zero.

`wells/clean` is 0 where the solver's convex optimisation failed and blanked
the whole well to NaN, 1 otherwise.

Reading in Julia: HDF5.jl reverses axes, so `permutedims(read(f["c"]))` etc.
"""

import glob
import os
import pickle
import sys

import numpy as np
import h5py

EQ_N = "dN_i/dt = g N_i [ sum_a (1-l) w c_ia R_a - m_i ]"
EQ_R = ("dR_a/dt = (R0_a-R_a)/tau - sum_i N_i c_ia R_a "
        "+ sum_b D_ab l R_b (sum_i N_i c_ib)")


def _scalar(params, key):
    a = np.asarray(params[key], dtype=float)
    if a.ndim and not np.allclose(a, a.flat[0]):
        raise ValueError("%r is not constant; the reduced equations in this "
                         "script assume scalar g, w, l, tau" % key)
    return float(a.flat[0]) if a.ndim else float(a)


def _strings(values):
    return np.array([str(v)[:32].encode("ascii", "replace") for v in values],
                    dtype="S32")


def convert(path, outdir):
    with open(path, "rb") as fh:
        N_df, R_df, params, R0_df, metadata = pickle.load(fh)

    N = np.asarray(N_df, dtype=float)      # (S_tot, n_wells) steady state
    R = np.asarray(R_df, dtype=float)      # (M, n_wells)     steady state
    S_tot, n_wells = N.shape
    M = R.shape[0]

    c = np.asarray(params["c"], dtype=float)
    D = np.asarray(params["D"], dtype=float)
    m = np.asarray(params["m"], dtype=float).reshape(-1)
    if c.shape != (S_tot, M) or D.shape != (M, M) or m.shape != (S_tot,):
        raise ValueError("unexpected shapes in %s" % path)

    # The EMP scripts add one scalar to a shared m per well; metadata['m'] is
    # the resulting mean, so the offset from well 0 (whose params are pickled)
    # is exact.  Elsewhere m is shared and the offset is zero.
    cols = list(getattr(metadata, "columns", []))
    if "m" in cols:
        m_mean = np.asarray(metadata["m"], dtype=float)
        m_offset = m_mean - m_mean[0]
    else:
        m_offset = np.zeros(n_wells)

    # params holds well 0 only.  Where params['R0'] disagrees with column 0 of
    # the stored table, that table is stale (true for Complex_environment_S150,
    # which kept MakeInitialState's single-source vector while the uniform
    # supply was actually integrated); params wins and is well-independent.
    R0 = np.asarray(R0_df, dtype=float)
    R0_params = np.asarray(params["R0"], dtype=float).reshape(-1)
    stale = not np.allclose(R0_params, R0[:, 0], rtol=1e-9, atol=1e-12)
    if stale:
        R0 = np.repeat(R0_params[:, None], n_wells, axis=1)

    # the solver blanks a whole well to NaN when its optimisation fails
    clean = ~(np.isnan(N).any(axis=0) | np.isnan(R).any(axis=0))

    dataset = os.path.basename(path)[len("comm_"):-len(".dat")]
    out = os.path.join(outdir, dataset + ".h5")
    comp = dict(compression="gzip", compression_opts=4)

    with h5py.File(out, "w") as h:
        a = h.attrs
        a["dataset"] = dataset
        a["source_file"] = os.path.abspath(path)
        a["generated_by"] = "extract_micrm_params.py"
        a["equation_N"] = EQ_N
        a["equation_R"] = EQ_R
        a["S_tot"], a["M"], a["n_wells"] = S_tot, M, n_wells
        a["g"] = _scalar(params, "g")
        a["w"] = _scalar(params, "w")
        a["l"] = _scalar(params, "l")
        a["tau"] = _scalar(params, "tau")
        a["m_reconstruction"] = "m_well_k = m + m_offset[k]"
        a["R0_table_was_stale"] = np.uint8(stale)
        a["array_layout"] = "c is (species, resource); R0/N/R are (index, well)"
        a["array_layout_julia"] = "HDF5.jl reverses axes; use permutedims"

        for key, val in (("c", c), ("D", D), ("m", m), ("m_offset", m_offset),
                         ("R0", R0), ("N", N), ("R", R)):
            h.create_dataset(key, data=val, **comp)

        w = h.create_group("wells")
        w.create_dataset("name", data=_strings(N_df.columns))
        w.create_dataset("clean", data=clean.astype(np.uint8))
        for col in cols:
            vals = metadata[col].values
            w.create_dataset(str(col).lower(),
                             data=_strings(vals) if vals.dtype.kind in "OSU"
                             else np.asarray(vals, dtype=float))

        if N_df.index.nlevels > 1 or R_df.index.nlevels > 1:
            g = h.create_group("labels")
            if N_df.index.nlevels > 1:
                g.create_dataset("species_family",
                                 data=_strings(N_df.index.get_level_values(0)))
            if R_df.index.nlevels > 1:
                g.create_dataset("resource_type",
                                 data=_strings(R_df.index.get_level_values(0)))

    print("  %-34s S_tot=%-5d M=%-4d wells=%-4d clean=%-4d%s"
          % (dataset, S_tot, M, n_wells, clean.sum(),
             "   [R0 table was stale]" if stale else ""))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "micrm_params_used")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    inputs = sorted(glob.glob(os.path.join(here, "data", "comm_*.dat")))
    print("writing to %s" % outdir)
    made = [convert(p, outdir) for p in inputs]
    print("\nwrote %d files, %.1f MB total"
          % (len(made), sum(os.path.getsize(p) for p in made) / 1e6))


if __name__ == "__main__":
    sys.exit(main())
