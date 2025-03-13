module SpaceMMiCRM

using Reexport
@reexport using ..SSMCMain
@reexport using ..SSMCMain.ModifiedMiCRM

using StatsBase

################################################################################
# Base
################################################################################
abstract type SingleAxisBC end
struct Periodic <: SingleAxisBC end
export SingleAxisBC, Periodic

abstract type AbstractSpace end

struct OneDCartesianSpace{B<:SingleAxisBC,F} <: AbstractSpace
    dx::F
    function OneDCartesianSpace(dx, _::B) where {B}
        new{B,typeof(dx)}(dx)
    end
end
export OneDCartesianSpace

function add_diffusion_single!(
    du::Vector,
    u::Vector,
    diffusion_constant,
    odc::OneDCartesianSpace{Periodic}
)
    ssize = length(u)
    for i in 1:ssize
        du[i] += diffusion_constant *
                 (u[mod1(i + 1, ssize)] - 2 * u[i] + u[mod1(i - 1, ssize)]) /
                 (odc.dx^2)
    end
end
"""
Adds the diffusive part of du for each field, the field index is taken to be
the first index, the others designating space.
"""
function add_diffusion!(
    du::AbstractMatrix,
    u::AbstractMatrix,
    diffusion_constants::AbstractVector,
    odc::OneDCartesianSpace{Periodic}
)
    ssize = size(u)[2]
    for i in 1:ssize
        du[:, i] .+= diffusion_constants .*
                     (u[:, mod1(i + 1, ssize)] .- 2 .* u[:, i] .+ u[:, mod1(i - 1, ssize)]) ./
                     (odc.dx^2)
    end
end
export add_diffusion_single!, add_diffusion!

"""Spatial Modified MiCRM model params"""
struct SMMiCRMParams{Ns,Nr,S<:AbstractSpace,F,N}
    mmicrm_params::MMiCRMParams{Ns,Nr,F}
    diffusion_constants::SVector{N,F}
    space::S

    function SMMiCRMParams(
        mmicrm_params::MMiCRMParams{Ns,Nr,F},
        diff::SVector{N,F},
        space::S
    ) where {Ns,Nr,S,F,N}
        if N == (Ns + Nr)
            new{Ns,Nr,S,F,N}(mmicrm_params, diff, space)
        else
            throw(ArgumentError("passed mmicrm_params and diff are not compatible"))
        end
    end
end
function smmicrmfunc!(du, u, p::SMMiCRMParams, t=0)
    for r in CartesianIndices(axes(u)[2:end])
        mmicrmfunc!((@view du[:, r]), (@view u[:, r]), p.mmicrm_params, t)
    end
    add_diffusion!(du, u, p.diffusion_constants, p.space)
    du
end
export SMMiCRMParams, smmicrmfunc!

################################################################################
# Smart functions
################################################################################
function make_smmicrmparams_smart(Ns, Nr, dx;
    bcs=Periodic(),
    diff_s=nothing,
    diff_r=nothing,
    kwargs...
)
    mmicrmparams = make_mmicrmparams_smart(Ns, Nr; kwargs...)
    diff_s = smart_val(diff_s, 0.0, Ns)
    diff_r = smart_val(diff_r, 0.0, Nr)
    diffs = SVector{Ns + Nr,Float64}(vcat(diff_s, diff_r))
    SMMiCRMParams(mmicrmparams, diffs, OneDCartesianSpace(dx, bcs))
end

function expand_u0_to_size(size, nospaceu0)
    u0 = Array{eltype(nospaceu0),1 + length(size)}(undef, length(nospaceu0), size...)
    for ci in CartesianIndices(size)
        u0[:, ci] .= nospaceu0
    end
    u0
end
export expand_u0_to_size

function make_smmicrmu0_smart(size, args...; kwargs...)
    nospaceu0 = make_mmicrmu0_smart(args...; kwargs...)

    expand_u0_to_size(size, nospaceu0)
end

"""Make the problem directly"""
function make_smmicrm_smart(Ns, Nr, size, dx, T=1.0;
    u0=:steadyR, u0rand=nothing, kwargs...
)
    params = make_smmicrmparams_smart(Ns, Nr, dx; kwargs...)
    u0 = make_smmicrmu0_smart(size, params.mmicrm_params; u0, u0rand)

    # make problem
    ODEProblem(smmicrmfunc!, u0, (0.0, T), params)
end
export make_smmicrm_smart, make_smmicrmparams_smart, make_smmicrmu0_smart

################################################################################
# Plotting
################################################################################
# function get_space_axes(size, odc::OneDCartesianSpace)
#     odc.dx .* (0:(size[1]-1))
# end
# export get_space_axes

function plot_smmicrm_sol_avgs(sol; singleax=false, plote=false)
    params = sol.prob.p
    if !isa(params, SMMiCRMParams)
        throw(ArgumentError("this func can only plot solutions of SMMiCRM problems"))
    end
    Ns, Nr = get_Ns(params.mmicrm_params)

    avgs = [mean(u, dims=2:ndims(u)) for u in sol.u]

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
        lines!(strainax, sol.t, getindex.(avgs, i); label=@sprintf "str %d" i)
    end
    for a in 1:Nr
        lines!(resax, sol.t, getindex.(avgs, Ns + a); label=@sprintf "res %d" a)
    end
    if plote
        # lines!(eax, sol.t, calc_E.(sol.u, Ref(params)); label=L"\epsilon")
    end

    if singleax
        axislegend(strainax)
    else
        axislegend(strainax)
        axislegend(resax)
        if plote
            axislegend(eax)
        end
    end
    fig
end
export plot_smmicrm_sol_avgs

function plot_1dsmmicrm_sol_snap(sol, t; singleax=false, plote=false)
    params = sol.prob.p
    if !isa(params, SMMiCRMParams)
        throw(ArgumentError("plot_smmicrm_sol_snap can only plot solutions of SMMiCRM problems"))
    end
    if !isa(params.space, OneDCartesianSpace)
        throw(ArgumentError("can only plot 1d thingys for now"))
    end
    Ns, Nr = get_Ns(params.mmicrm_params)

    len = size(sol.u[1])[2]
    xs = params.space.dx .* (0:(len-1))

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
        lines!(strainax, xs, sol(t)[i, :]; label=@sprintf "str %d" i)
    end
    for a in 1:Nr
        lines!(resax, xs, sol(t)[Ns+a, :]; label=@sprintf "res %d" a)
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
