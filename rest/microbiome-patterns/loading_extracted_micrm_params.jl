"""
loading_extracted_micrm_params.jl

Load the files written by `extract_micrm_params.py` into `micrm_params_used/`
and build `BMMiCRMParams` from them.

Source data: Marsland III, Cui & Mehta, Sci. Rep. 10:3308 (2020).

Parameter mapping, paper -> ModifiedMiCRM:

    g, w        scalars in every dataset, broadcast to vectors
    m[i]        m .+ m_offset[k]
    K[a]      = R0[a] / tau        supply rate
    r[a]      = 1 / tau            dilution rate
    l[i,a]    = l                  scalar leakage, broadcast over strains
    c[i,a]    = c[i,a]
    D[i,a,b]  = D[a,b]             same b -> a convention, shared by all strains

No filtering and no options: every well and every strain is kept. `clean` is a
column, false where the solver failed and blanked the well to NaN - those rows
have NaN in `u_ss`, so filter with `df[df.clean, :]` before doing anything
numerical. Extinction thresholds are yours to pick downstream; the seeded
subpool of well k is exactly `f.N[:, k] .> 0`.

Memory: `BMMiCRMParams` stores D densely as (Ns, Nr, Nr), so the full pool
costs Ns*Nr^2*8 bytes - 12 MB for the EMP datasets but 3.6 GB for the HMP ones.
`micrm_dataframe` builds it once and shares it across rows, so that is one copy
per file, not one per well. Subset the strains yourself if that is too much.
"""

using SSMCMain, SSMCMain.ModifiedMiCRM
using HDF5
using DataFrames

# HDF5 and ModifiedMiCRM both export `attributes`, so it is qualified below

struct MiCRMFile
    dataset::String
    c::Matrix{Float64}          # (Ns, Nr)
    D::Matrix{Float64}          # (Nr, Nr), b -> a
    m::Vector{Float64}          # (Ns,)
    m_offset::Vector{Float64}   # (n_wells,)
    R0::Matrix{Float64}         # (Nr, n_wells)
    N::Matrix{Float64}          # (Ns, n_wells)  steady state
    R::Matrix{Float64}          # (Nr, n_wells)  steady state
    g::Float64
    w::Float64
    l::Float64
    tau::Float64
    wells::Vector{String}
    clean::Vector{Bool}
    meta::Dict{String,Any}      # the dataset's own per-well columns
end

"""
    load_micrm_file(path) -> MiCRMFile

HDF5.jl returns arrays with reversed axes, so they are permuted back here.
"""
function load_micrm_file(path::AbstractString)
    h5open(path) do f
        meta = Dict{String,Any}()
        for k in keys(f["wells"])
            k in ("name", "clean") || (meta[k] = read(f["wells"][k]))
        end
        MiCRMFile(
            read(HDF5.attributes(f)["dataset"]),
            permutedims(read(f["c"])), permutedims(read(f["D"])),
            read(f["m"]), read(f["m_offset"]),
            permutedims(read(f["R0"])),
            permutedims(read(f["N"])), permutedims(read(f["R"])),
            read(HDF5.attributes(f)["g"]), read(HDF5.attributes(f)["w"]),
            read(HDF5.attributes(f)["l"]), read(HDF5.attributes(f)["tau"]),
            read(f["wells/name"]), read(f["wells/clean"]) .== 1,
            meta,
        )
    end
end

"""
    bmmicrm_params(f, k) -> BMMiCRMParams

Parameters for well `k`, whole regional pool.
"""
function bmmicrm_params(f::MiCRMFile, k::Integer)
    Ns, Nr = size(f.c)
    D3 = Array{Float64,3}(undef, Ns, Nr, Nr)
    @inbounds for i in 1:Ns
        D3[i, :, :] .= f.D
    end
    BMMiCRMParams(fill(f.g, Ns), fill(f.w, Nr),
        f.m .+ f.m_offset[k], f.R0[:, k] ./ f.tau, fill(1 / f.tau, Nr),
        fill(f.l, Ns, Nr), f.c, D3)
end

"""
    steady_state(f, k) -> Vector

The saved steady state for well `k`, stacked as `[N; R]` the way
`mmicrmfunc!` expects.
"""
steady_state(f::MiCRMFile, k::Integer) = vcat(f.N[:, k], f.R[:, k])

"""
    micrm_dataframe(path_or_file) -> DataFrame

One row per well: `well`, `clean`, the dataset's own metadata columns,
`params` and `u_ss`. The arrays that do not depend on the well are built once
and shared by every row, so do not mutate them in place.
"""
function micrm_dataframe(f::MiCRMFile)
    Ns, Nr = size(f.c)
    D3 = Array{Float64,3}(undef, Ns, Nr, Nr)
    @inbounds for i in 1:Ns
        D3[i, :, :] .= f.D
    end
    gv, wv = fill(f.g, Ns), fill(f.w, Nr)
    rv, lmat = fill(1 / f.tau, Nr), fill(f.l, Ns, Nr)

    df = DataFrame(well=f.wells, clean=f.clean)
    for key in sort(collect(keys(f.meta)))
        df[!, Symbol(key)] = f.meta[key]
    end
    df.params = [BMMiCRMParams(gv, wv, f.m .+ f.m_offset[k],
                     f.R0[:, k] ./ f.tau, rv, lmat, f.c, D3)
                 for k in eachindex(f.wells)]
    df.u_ss = [vcat(f.N[:, k], f.R[:, k]) for k in eachindex(f.wells)]
    df
end

micrm_dataframe(path::AbstractString) = micrm_dataframe(load_micrm_file(path))
