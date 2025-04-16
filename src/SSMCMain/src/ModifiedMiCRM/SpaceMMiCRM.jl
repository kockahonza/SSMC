module SpaceMMiCRM

using Reexport
@reexport using ..SSMCMain
@reexport using ..SSMCMain.ModifiedMiCRM
import ..SSMCMain.ModifiedMiCRM: get_Ns

using StatsBase
using Interpolations

using ADTypes, SparseConnectivityTracer

using Base.Threads
using ChunkSplitters

import Base: ndims

################################################################################
# Base
################################################################################
# The abstrat space interface - the only thing a space should do is diffusion
abstract type AbstractSpace end
function ndims(s::AbstractSpace)
    throw(ErrorException(@sprintf "no function ndims defined for space type %s" string(typeof(s))))
end
"""
Adds the diffusive part of du for each field, the field index is taken to be
the first index, the others designating space.
"""
function add_diffusion!(du, u, diffusion_constants, s::AbstractSpace, usenthreads=nothing)
    throw(ErrorException(@sprintf "no function add_diffusion! defined for space type %s" string(typeof(s))))
end
export add_diffusion!

abstract type SingleAxisBC end
struct Periodic <: SingleAxisBC end
struct Closed <: SingleAxisBC end
export SingleAxisBC, Periodic, Closed

# The main Spatial Modified MiCRM params (which contains and AbstractSpace) and logic
"""Spatial Modified MiCRM model params"""
struct SMMiCRMParams{Ns,Nr,S<:AbstractSpace,F,N,P<:Union{Nothing,Int},A,B}
    mmicrm_params::MMiCRMParams{Ns,Nr,F,A,B}
    diffusion_constants::SVector{N,F}
    space::S

    usenthreads::P

    function SMMiCRMParams(
        mmicrm_params::MMiCRMParams{Ns,Nr,F,A,B},
        diff::SVector{N,F},
        space::S,
        usenthreads=nothing
    ) where {Ns,Nr,S,F,N,A,B}
        if N == (Ns + Nr)
            new{Ns,Nr,S,F,N,typeof(usenthreads),A,B}(mmicrm_params, diff, space, usenthreads)
        else
            throw(ArgumentError("passed mmicrm_params and diff are not compatible"))
        end
    end
end
get_Ns(sp::SMMiCRMParams) = get_Ns(sp.mmicrm_params)
ndims(sp::SMMiCRMParams) = ndims(sp.space)
export SMMiCRMParams

function smmicrmfunc!(du, u, p::SMMiCRMParams, t=0)
    if isnothing(p.usenthreads)
        @inbounds for r in CartesianIndices(axes(u)[2:end])
            mmicrmfunc!((@view du[:, r]), (@view u[:, r]), p.mmicrm_params, t)
        end
    else
        rchunks = chunks(CartesianIndices(axes(u)[2:end]); n=p.usenthreads)
        @sync for rs in rchunks
            @spawn begin
                @inbounds for r in rs
                    mmicrmfunc!((@view du[:, r]), (@view u[:, r]), p.mmicrm_params, t)
                end
            end
        end
    end
    add_diffusion!(du, u, p.diffusion_constants, p.space, p.usenthreads)
    nothing
end
export smmicrmfunc!

# Makes an ODEProblem checking u0 and SMMiCRMParams are compatible
function make_smmicrm_problem_safe(u0, T, smmicrm_params; sparse_jac=true, t0=0.0)
    u0ndims = ndims(u0) - 1
    spacendims = ndims(smmicrm_params.space)
    if u0ndims != spacendims
        throw(ArgumentError(@sprintf "the passed u0 and space do not have the same number of dimensions, u0ndims is %d and ndims(space) is %d" u0ndims spacendims))
    end
    if any(s -> s < 4, size(u0)[2:end])
        throw(ArgumentError("getting u0 that has too few points for the optimized code to work"))
    end

    func = if sparse_jac
        jac_prot = ADTypes.jacobian_sparsity(
            (du, u) -> smmicrmfunc!(du, u, smmicrm_params),
            similar(u0),
            u0,
            TracerSparsityDetector()
        )
        ODEFunction(smmicrmfunc!; jac_prototype=float.(jac_prot))
    else
        smmicrmfunc!
    end

    ODEProblem(func, u0, (t0, t0 + T), smmicrm_params)
end
function make_smmicrm_problem_safe(u0, T, args...; kwargs...)
    make_smmicrm_problem_safe(u0, T, SMMiCRMParams(args...); kwargs...)
end
export make_smmicrm_problem_safe

function change_smmicrm_params(sp::SMMiCRMParams;
    mmicrm_params=sp.mmicrm_params,
    diffusion_constants=sp.diffusion_constants,
    space=sp.space,
    usenthreads=sp.usenthreads
)
    SMMiCRMParams(mmicrm_params, diffusion_constants, space, usenthreads)
end
export change_smmicrm_params

################################################################################
# Different AbstractSpace implementations
################################################################################
include("smmicrm_spaces.jl")

################################################################################
# Going from MMiCRM to SMMiCRM
################################################################################
function add_space_to_mmicrmparams(mmicrm_params::MMiCRMParams, D;
    diff_s=nothing,
    diff_r=nothing,
    dx=nothing,
    bcs=Periodic(),
    usenthreads=nothing
)
    Ns, Nr = get_Ns(mmicrm_params)
    diff_s = smart_val(diff_s, 0.0, Ns)
    diff_r = smart_val(diff_r, 0.0, Nr)
    diffs = SVector{Ns + Nr,Float64}(vcat(diff_s, diff_r))

    space = make_cartesianspace_smart(D; dx, bcs)

    SMMiCRMParams(mmicrm_params, diffs, space, usenthreads)
end
export add_space_to_mmicrmparams

function expand_u0_to_size(size::Tuple, nospaceu0)
    u0 = Array{eltype(nospaceu0),1 + length(size)}(undef, length(nospaceu0), size...)
    for ci in CartesianIndices(size)
        u0[:, ci] .= nospaceu0
    end
    u0
end
export expand_u0_to_size

function add_space_to_mmicrm(p::ODEProblem, u0, diffusion_constants, T, size, dx, bcs=Periodic())
    if !isa(p.p, MMiCRMParams)
        throw(ArgumentError("passed problem is not a Modified CRM problem"))
    end

    su0 = expand_u0_to_size(size, u0)
    space = make_cartesianspace_smart(length(size); dx, bcs)

    make_smmicrm_problem_safe(su0, T, p.p, diffusion_constants, space)
end
export add_space_to_mmicrm

################################################################################
# Problem remaking/changing params
################################################################################
function perturb_u0_uniform(Ns, Nr, u0, e_s=nothing, e_r=nothing)
    N = Ns + Nr

    epsilon = if isnothing(e_r)
        smart_val(e_s, nothing, N)
    else
        vcat(smart_val(e_s, nothing, Ns), smart_val(e_r, nothing, Nr))
    end

    rand_pm_one = (2 .* rand(size(u0)...) .- 1)

    u0 .+ rand_pm_one .* epsilon
end
function perturb_u0_uniform_prop(Ns, Nr, u0, e_s=nothing, e_r=nothing)
    N = Ns + Nr

    epsilon = if isnothing(e_r)
        smart_val(e_s, nothing, N)
    else
        vcat(smart_val(e_s, nothing, Ns), smart_val(e_r, nothing, Nr))
    end

    rand_pm_one = (2 .* rand(size(u0)...) .- 1)

    u0 .* (1.0 .+ rand_pm_one .* epsilon)
end
function perturb_u0_wave(Ns, Nr, u0, args...)
    # TODO: Implement this!
end
export perturb_u0_uniform, perturb_u0_uniform_prop, perturb_u0_wave

function perturb_smmicrm_u0(p::ODEProblem, ptype, args...)
    if !isa(p.p, SMMiCRMParams)
        throw(ArgumentError("passed problem is not a Spatial Modified CRM problem"))
    end
    if ptype == :uniform
        pu0 = perturb_u0_uniform(get_Ns(p.p)..., p.u0, args...)
    else
        throw(ArgumentError("unrecognised ptype"))
    end

    remake(p; u0=pu0)
end
export perturb_smmicrm_u0

function change_usenthreads(p::ODEProblem, usenthreads)
    if !isa(p.p, SMMiCRMParams)
        throw(ArgumentError("passed problem is not a Spatial Modified CRM problem"))
    end
    newp = SMMiCRMParams(p.p.mmicrm_params, p.p.diffusion_constants, p.p.space, usenthreads)

    remake(p; p=newp)
end
export change_usenthreads

################################################################################
# Smart functions
################################################################################
function make_smmicrmparams_smart(Ns, Nr, D;
    diff_s=nothing, diff_r=nothing, dx=nothing, bcs=Periodic(),
    kwargs...
)
    mmicrm_params = make_mmicrmparams_smart(Ns, Nr; kwargs...)

    add_space_to_mmicrmparams(mmicrm_params, D; diff_s, diff_r, dx, bcs)
end

function make_smmicrmu0_smart(size, args...; kwargs...)
    nospaceu0 = make_mmicrmu0_smart(args...; kwargs...)

    expand_u0_to_size(size, nospaceu0)
end

"""Make the problem directly"""
function make_smmicrm_smart(Ns, Nr, size, T=1.0;
    u0=:steadyR, u0rand=nothing, kwargs...
)
    sp = make_smmicrmparams_smart(Ns, Nr, length(size); kwargs...)
    u0 = make_smmicrmu0_smart(size, sp.mmicrm_params; u0, u0rand)

    # make problem
    make_smmicrm_problem_safe(u0, T, sp)
end
export make_smmicrmparams_smart, make_smmicrmu0_smart, make_smmicrm_smart

################################################################################
# Plotting
################################################################################
include("smmicrm_plotting.jl")

include("smmicrm_util.jl")

end
