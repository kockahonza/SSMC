"""Modified MiCRM model"""
module ModifiedMiCRM

using Reexport
@reexport using ..SSMCMain

using Base.Threads
using OhMyThreads
using ChunkSplitters

using StatsBase
using Interpolations
using Polynomials

using ADTypes, SparseConnectivityTracer

using GraphvizDotLang: digraph, subgraph, node, edge, attr, save
using Colors
using ColorSchemes

import Base: ndims, getproperty, copy

################################################################################
# Nospace Modified MiCRM model params types interface specification
################################################################################
abstract type AbstractMMiCRMParams{F} end
function get_Ns(p::AbstractMMiCRMParams)
    throw(ArgumentError(@sprintf "get_Ns not implemented for AbstractMMiCRMParams type %s" string(typeof(p))))
end
function mmicrmfunc!(du, u, p::AbstractMMiCRMParams, t=0)
    throw(ArgumentError(@sprintf "mmicrmfunc! not implemented for AbstractMMiCRMParams type %s" string(typeof(p))))
end
# getproperty should also be overloaded for g, w, m, K, r, l, c and D
export AbstractMMiCRMParams, get_Ns, mmicrmfunc!

################################################################################
# Similar interface for spatial MMiCRM model params
################################################################################
# Abstract space interface - the only thing a space should do is diffusion
abstract type AbstractSpace end
function ndims(s::AbstractSpace)
    throw(ErrorException(@sprintf "no function ndims defined for space type %s" string(typeof(s))))
end
"""
Adds the diffusive part of du for each field, the field index is taken to be
the first index, the others designating space.
"""
function add_diffusion!(du, u, Ds, s::AbstractSpace, usenthreads=nothing)
    throw(ErrorException(@sprintf "no function add_diffusion! defined for space type %s" string(typeof(s))))
end
function space_cell_size(s::AbstractSpace)
    throw(ErrorException(@sprintf "no function space_cell_size defined for space type %s" string(typeof(s))))
end
export AbstractSpace, space_cell_size, add_diffusion!

abstract type SingleAxisBC end
struct Periodic <: SingleAxisBC end
struct Closed <: SingleAxisBC end
export SingleAxisBC, Periodic, Closed

# Abstract Spatial MMiCRM params interface, there are two subclasses here:
# - is S is Nothing than these are just the params, aka nospace params plus diffusions
# - if S is an AbstractSpace there is extra information on what type of space it is
abstract type AbstractSMMiCRMParams{S<:Union{Nothing,AbstractSpace},F} <: AbstractMMiCRMParams{F} end
function get_Ds(p::AbstractSMMiCRMParams)
    throw(ArgumentError(@sprintf "get_Ds not implemented for AbstractSMMiCRMParams type %s" string(typeof(p))))
end
function get_space(sp::AbstractSMMiCRMParams{S}) where {S}
    if isnothing(S)
        return nothing
    else
        throw(ArgumentError(@sprintf "get_space not implemented for AbstractSMMiCRMParams type %s" string(typeof(sp))))
    end
end
ndims(sp::AbstractSMMiCRMParams) = ndims(get_space(sp))
function smmicrmfunc!(du, u, sp::AbstractSMMiCRMParams, t=0)
    throw(ArgumentError(@sprintf "smmicrmfunc! not implemented for AbstractSMMiCRMParams type %s" string(typeof(sp))))
end
export AbstractSMMiCRMParams, get_Ds, get_space, smmicrmfunc!

function check_mmicrmparams(p::AbstractMMiCRMParams;
    threshold=10 * eps()
)
    Ns, Nr = get_Ns(p)
    # check all the array sizes
    if size(p.g) != (Ns,)
        @error (@sprintf "g has wrong size %s, should be %s" string(size(p.g)) string((Ns,)))
    end
    if size(p.w) != (Nr,)
        @error (@sprintf "w has wrong size %s, should be %s" string(size(p.w)) string((Nr,)))
    end
    if size(p.m) != (Ns,)
        @error (@sprintf "m has wrong size %s, should be %s" string(size(p.m)) string((Ns,)))
    end
    if size(p.K) != (Nr,)
        @error (@sprintf "w has wrong size %s, should be %s" string(size(p.K)) string((Nr,)))
    end
    if size(p.r) != (Nr,)
        @error (@sprintf "w has wrong size %s, should be %s" string(size(p.r)) string((Nr,)))
    end
    if size(p.l) != (Ns, Nr)
        @error (@sprintf "l has wrong size %s, should be %s" string(size(p.l)) string((Ns, Nr)))
    end
    if size(p.c) != (Ns, Nr)
        @error (@sprintf "l has wrong size %s, should be %s" string(size(p.c)) string((Ns, Nr)))
    end
    if size(p.D) != (Ns, Nr, Nr)
        @error (@sprintf "D has wrong size %s, should be %s" string(size(p.D)) string((Ns, Nr, Nr)))
    end

    # check D
    for i in 1:Ns
        for a in 1:Nr
            total_out = 0.0
            for b in 1:Nr
                total_out += p.D[i, b, a]
            end
            if (total_out - 1.0) > threshold
                @error (@sprintf "strain %d leaks more than it energetically can through consuming %d (total_out is %g)" i a total_out)
            end
            if ((1.0 - total_out) > threshold) && (p.l[i, a] != 0)
                @info (@sprintf "strain %d leaks less than it energetically can through consuming %d (total_out is %g)" i a total_out)
            end
        end
    end
end
export check_mmicrmparams

# Making ODEProblem s
function make_mmicrm_problem(p::AbstractMMiCRMParams, u0, T;
    sparse_jac=false, t0=0.0, kwargs...
)
    if (ndims(u0) != 1) || (length(u0) != sum(get_Ns(p)))
        throw(ArgumentError(
            @sprintf "passed u0 is not compatible with passed params, size(u0) is %s and Ns, Nr are %s" string(size(u0)) string(get_Ns(p))
        ))
    end

    func = if sparse_jac
        jac_prot = ADTypes.jacobian_sparsity(
            (du, u) -> mmicrmfunc!(du, u, p),
            similar(u0),
            u0,
            TracerSparsityDetector()
        )
        ODEFunction(mmicrmfunc!; jac_prototype=float.(jac_prot))
    else
        mmicrmfunc!
    end

    ODEProblem(func, u0, (t0, t0 + T), p; kwargs...)
end
export make_mmicrm_problem

function make_mmicrm_ss_problem(p::AbstractMMiCRMParams, u0;
    sparse_jac=false, kwargs...
)
    if (ndims(u0) != 1) || (length(u0) != sum(get_Ns(p)))
        throw(ArgumentError(
            @sprintf "passed u0 is not compatible with passed params, size(u0) is %s and Ns, Nr are %s" string(size(u0)) string(get_Ns(p))
        ))
    end

    func = if sparse_jac
        jac_prot = ADTypes.jacobian_sparsity(
            (du, u) -> mmicrmfunc!(du, u, p),
            similar(u0),
            u0,
            TracerSparsityDetector()
        )
        ODEFunction(mmicrmfunc!; jac_prototype=float.(jac_prot))
    else
        mmicrmfunc!
    end

    SteadyStateProblem(func, u0, p; kwargs...)
end
export make_mmicrm_ss_problem

function make_smmicrm_problem(p::AbstractSMMiCRMParams, u0, T;
    sparse_jac=true, t0=0.0, kwargs...
)
    # check the Ns, Nr match up
    if size(u0)[1] != sum(get_Ns(p))
        throw(ArgumentError(
            @sprintf "passed u0 is not compatible with passed params, size(u0) is %s and Ns, Nr are %s" string(size(u0)) string(get_Ns(p))
        ))
    end

    # check u0 is compatible with the space
    u0ndims = ndims(u0) - 1
    spacendims = ndims(get_space(p))
    if u0ndims != spacendims
        throw(ArgumentError(@sprintf "the passed u0 and space do not have the same number of dimensions, u0ndims is %d and ndims(space) is %d" u0ndims spacendims))
    end
    if any(s -> s < 4, size(u0)[2:end])
        throw(ArgumentError("u0 spatial size is so small that there may be numerical issues from violated assumption"))
    end

    func = if sparse_jac
        jac_prot = ADTypes.jacobian_sparsity(
            (du, u) -> smmicrmfunc!(du, u, p),
            similar(u0),
            u0,
            TracerSparsityDetector()
        )
        ODEFunction(smmicrmfunc!; jac_prototype=float.(jac_prot))
    else
        smmicrmfunc!
    end

    ODEProblem(func, u0, (t0, t0 + T), p; kwargs...)
end
export make_smmicrm_problem

function mmicrmresid(u, p::AbstractMMiCRMParams)
    du = zeros(eltype(u), size(u))
    mmicrmfunc!(du, u, p)
    du
end
function mmicrmresid(s::ODESolution)
    mmicrmresid(s.u[end], s.prob.p)
end
function mmicrmmaxresid(args...)
    maximum(abs, mmicrmresid(args...))
end
export mmicrmresid, mmicrmmaxresid

################################################################################
# Imports
################################################################################
# Cartesian spaces - these do diffusion
include("cartesian_space.jl")

# Setups up initial states
include("u0_prep.jl")

# Implementations of the MMiCRMParams interfaces
include("params_staticarray.jl")
include("params_basearray.jl")

# Linear stability analysis
include("linstab/linstab.jl")

# Plotting
include("plotting.jl")

# Physicsy bits
include("physics.jl")
include("dimensional_analysis.jl")

include("GraphAnalysis.jl")

include("util.jl")
include("diagrams.jl")

################################################################################
# Sub/optional bits that come in submodules
################################################################################
# Minimal model specific bits
include("MinimalModelSemisymbolic/MinimalModelSemisymbolic.jl")
include("MinimalModelSemisymbolic/MinimalModelV2.jl")
include("RandomSystems/RandomSystems.jl")

end
