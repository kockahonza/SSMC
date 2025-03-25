module SpaceMMiCRM

using Reexport
@reexport using ..SSMCMain
@reexport using ..SSMCMain.ModifiedMiCRM
import ..SSMCMain.ModifiedMiCRM: get_Ns

using StatsBase

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
        rchunks = chunks(CartesianIndices(axes(u)[2:end]), p.usenthreads)
        @sync for (rs, _) in rchunks
            @spawn begin
                @inbounds for r in rs
                    mmicrmfunc!((@view du[:, r]), (@view u[:, r]), p.mmicrm_params, t)
                end
            end
        end
    end
    add_diffusion!(du, u, p.diffusion_constants, p.space, p.usenthreads)
    du
end
export smmicrmfunc!

# Makes an ODEProblem checking u0 and SMMiCRMParams are compatible
function make_smmicrm_problem_safe(u0, T, smmicrm_params; use_sa=true)
    u0ndims = ndims(u0) - 1
    spacendims = ndims(smmicrm_params.space)
    if u0ndims != spacendims
        throw(ArgumentError(@sprintf "the passed u0 and space do not have the same number of dimensions, u0ndims is %d and ndims(space) is %d" u0ndims spacendims))
    end
    u0Nsr = size(u0)[1]
    ModifiedMiCRM.get_Ns(smmicrm_params.mmicrm_params)

    if use_sa
        u0 = SizedArray{Tuple{size(u0)...}}(u0)
    end

    ODEProblem(smmicrmfunc!, u0, (0.0, T), smmicrm_params)
end
function make_smmicrm_problem_safe(u0, T, args...; kwargs...)
    make_smmicrm_problem_safe(u0, T, SMMiCRMParams(args...); kwargs...)
end
export make_smmicrm_problem_safe

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

    u0 .+ rand(size(u0)...) .* epsilon
end
function perturb_u0_wave(Ns, Nr, u0, args...)
    # TODO: Implement this!
end
export perturb_u0_uniform, perturb_u0_wave

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
# function plot_smmicrm_sol_avgs(sol; singleax=false, plote=false)
#     params = sol.prob.p
#     if !isa(params, SMMiCRMParams)
#         throw(ArgumentError("this func can only plot solutions of SMMiCRM problems"))
#     end
#     Ns, Nr = get_Ns(params.mmicrm_params)
#
#     avgs = [mean(u, dims=2:ndims(u)) for u in sol.u]
#
#     fig = Figure()
#     if singleax
#         strainax = resax = Axis(fig[1, 1])
#         if plote
#             eax = strainax
#         end
#     else
#         strainax = Axis(fig[1, 1])
#         resax = Axis(fig[2, 1])
#         if plote
#             eax = Axis(fig[3, 1])
#         end
#     end
#
#     # plot data
#     for i in 1:Ns
#         lines!(strainax, sol.t, getindex.(avgs, i); label=@sprintf "str %d" i)
#     end
#     for a in 1:Nr
#         lines!(resax, sol.t, getindex.(avgs, Ns + a); label=@sprintf "res %d" a)
#     end
#     if plote
#         # lines!(eax, sol.t, calc_E.(sol.u, Ref(params)); label=L"\epsilon")
#     end
#
#     if singleax
#         axislegend(strainax)
#     else
#         axislegend(strainax)
#         axislegend(resax)
#         if plote
#             axislegend(eax)
#         end
#     end
#     fig
# end
# export plot_smmicrm_sol_avgs

function plot_1dsmmicrm_sol_snap(params, snap_u; singleax=false, plote=false)
    if !isa(params, SMMiCRMParams) || (ndims(params) != 1)
        throw(ArgumentError("plot_smmicrm_sol_snap can only plot 1D solutions of SMMiCRM problems"))
    end
    Ns, Nr = get_Ns(params.mmicrm_params)

    len = size(snap_u)[2]
    xs = params.space.dx[1] .* (0:(len-1))

    fig = Figure()
    if singleax
        strainax = resax = Axis(fig[1, 1])
        if plote
            eax = strainax
        end
    else
        strainax = Axis(fig[1, 1])
        resax = Axis(fig[2, 1])
        if plote
            eax = Axis(fig[3, 1])
        end
    end

    # plot data
    for i in 1:Ns
        lines!(strainax, xs, snap_u[i, :]; label=@sprintf "str %d" i)
    end
    for a in 1:Nr
        lines!(resax, xs, snap_u[Ns+a, :]; label=@sprintf "res %d" a)
    end
    # if plote
    #     lines!(eax, xs, calc_E.(sol.u, Ref(params)); label=L"\epsilon")
    # end

    if singleax
        axislegend(strainax)
    else
        axislegend(strainax)
        axislegend(resax)
        if plote
            axislegend(eax)
        end
    end

    if singleax
        axs = strainax
    else
        axs = [strainax, resax]
    end

    FigureAxisAnything(fig, axs, nothing)
end
export plot_1dsmmicrm_sol_snap

end
