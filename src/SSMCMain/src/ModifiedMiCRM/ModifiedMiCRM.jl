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
function mmicrmjac!(J, u, p::AbstractMMiCRMParams, t=0)
    throw(ArgumentError(@sprintf "mmicrmjac! not implemented for AbstractMMiCRMParams type %s" string(typeof(p))))
end
# getproperty should also be overloaded for g, w, m, K, r, l, c and D
export AbstractMMiCRMParams, get_Ns, mmicrmfunc!, mmicrmjac!

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
function add_diffusion_jac!(J, u, Ds, s::AbstractSpace, usenthreads=nothing)
    throw(ErrorException(@sprintf "no function add_diffusion_jac! defined for space type %s" string(typeof(s))))
end
function space_cell_size(s::AbstractSpace)
    throw(ErrorException(@sprintf "no function space_cell_size defined for space type %s" string(typeof(s))))
end
export AbstractSpace, space_cell_size, add_diffusion!, add_diffusion_jac!

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
function smmicrmjac!(J, u, sp::AbstractSMMiCRMParams, t=0)
    throw(ArgumentError(@sprintf "smmicrmjac! not implemented for AbstractSMMiCRMParams type %s" string(typeof(sp))))
end
export AbstractSMMiCRMParams, get_Ds, get_space, smmicrmfunc!, smmicrmjac!

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
    usejac=true, t0=0.0, kwargs...
)
    if (ndims(u0) != 1) || (length(u0) != sum(get_Ns(p)))
        throw(ArgumentError(
            @sprintf "passed u0 is not compatible with passed params, size(u0) is %s and Ns, Nr are %s" string(size(u0)) string(get_Ns(p))
        ))
    end

    func = if usejac
        ODEFunction(mmicrmfunc!; jac=mmicrmjac!)
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
    jac_type=:both, t0=0.0, kwargs...
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

    func = if jac_type == :sparse
        jac_prot = ADTypes.jacobian_sparsity(
            (du, u) -> smmicrmfunc!(du, u, p),
            similar(u0),
            u0,
            TracerSparsityDetector()
        )
        ODEFunction(smmicrmfunc!; jac_prototype=float.(jac_prot))
    elseif jac_type == :predef
        ODEFunction(smmicrmfunc!; jac=smmicrmjac!)
    elseif jac_type == :both
        jac_prot = ADTypes.jacobian_sparsity(
            (du, u) -> smmicrmfunc!(du, u, p),
            similar(u0),
            u0,
            TracerSparsityDetector()
        )
        ODEFunction(smmicrmfunc!;
            jac_prototype=float.(jac_prot),
            jac=smmicrmjac!,
        )
    elseif jac_type == :none
        smmicrmfunc!
    else
        throw(ArgumentError(@sprintf "jac_type %s not recognized, should be one of :sparse, :predef, :both or :none" string(jac_type)))
    end

    ODEProblem(func, u0, (t0, t0 + T), p; kwargs...)
end
export make_smmicrm_problem

function mmicrmresid(u, p::AbstractMMiCRMParams)
    @assert ndims(u) == 1
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

function smmicrmresid(u, p::AbstractSMMiCRMParams)
    @assert ndims(u) > 1
    du = zeros(eltype(u), size(u))
    smmicrmfunc!(du, u, p)
    du
end
function smmicrmresid(s::ODESolution)
    smmicrmresid(s.u[end], s.prob.p)
end
function smmicrmmaxresid(args...)
    maximum(abs, smmicrmresid(args...))
end
export smmicrmresid, smmicrmmaxresid

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

module LeakageScale
using Printf

e(x) = 1 / (1 + exp(x))
l(x) = 1 - e(x)
ltox(l) = log(l / (1 - l))
etox(e) = -log(e / (1 - e))

exticks(es) = (etox.(es), [@sprintf("%.2f", e) for e in es])
lxticks(ls) = (ltox.(ls), [@sprintf("%.2f", l) for l in ls])

"""
`n` minor ticks between each pair of neighbouring major ticks, spaced linearly in
leakage so they bunch up where the scale compresses, showing how it distorts
space. Pass the same values used for the major ticks, along with
`yminorticksvisible=true`, eg `yminorticks=LeakageScale.lxminorticks(lticks, 4)`.
"""
lxminorticks(ls, n=9) = [ltox(x) for (lo, hi) in zip(ls[1:(end-1)], ls[2:end])
                         for x in range(lo, hi, n + 2)[2:(end-1)]]
exminorticks(es, n=9) = lxminorticks(1 .- es, n)

function lxrange(lmin, lmax, n)
    l.(range(ltox(lmin), ltox(lmax), n))
end
lxrange(deltal, n) = lxrange(deltal, 1 - deltal, n)

end
export LeakageScale

module CustomSymlogScale
using Makie
using Printf

"""
    scale(linthresh)

Symlog-style Makie scale, `y -> sign(y) * log10(1 + |y|/linthresh)`, so both signs
and many decades of magnitude fit on one axis. Note this is one smooth function
rather than a piecewise linear-then-log join, so it is only *asymptotically*
linear below `linthresh` and *asymptotically* one unit of axis space per factor of
10 above it: at `linthresh=1` the core `0 -> 1` spans 0.301 units with its ends
stretched 1.9x relative to each other, and the decades above it get 0.740, 0.963,
0.996, 0.9996 units. Pair with `ticks`/`MinorTicks` (or just use `kwargs`) -
Makie's automatic ticks fall back to linearly spaced values for any non-log scale,
which is what makes a bare symlog axis look wrong.
"""
function scale(linthresh)
    t = float(linthresh)
    t > 0. || throw(ArgumentError("linthresh must be positive"))
    forward = y -> sign(y) * log10(1. + abs(y) / t)
    inverse = z -> sign(z) * t * (exp10(abs(z)) - 1.)
    Makie.ReversibleScale(forward, inverse; limits=(-3f0, 3f0), name=:symlog10)
end

"""
    ticklabel(v)

`0` as itself, powers of 10 as `10^n`, anything else as a plain `%.3g` number.
"""
function ticklabel(v)
    v == 0. && return rich("0")
    sgn = v < 0. ? Makie.MINUS_SIGN : ""
    e = log10(abs(v))
    isapprox(e, round(e); atol=1e-9) || return rich(sgn * (@sprintf "%.3g" abs(v)))
    rich(sgn * "10", superscript(replace(string(round(Int, e)), "-" => Makie.MINUS_SIGN);
        offset=Vec2f(0.1, 0.)))
end

"""
    ticks(vals)

Major ticks at exactly `vals`, labelled by `ticklabel`. Use when you want them
somewhere the `linthresh` method cannot reach - it steps whole decades, so it
never places more than one tick per decade.
"""
ticks(vals::AbstractVector) = (vals, ticklabel.(vals))

"""
    ticks(linthresh; max_ticks=11, trim=0.)

Ticks at 0 and ±10^n from `linthresh` up, thinned to at most `max_ticks`. Nonzero
ticks with `|v| <= trim * linthresh` are dropped, for when the ones closest to the
linear region crowd each other.
"""
function ticks(linthresh::Real; max_ticks=11, trim=0.)
    e0 = round(Int, log10(float(linthresh)))
    (vmin, vmax) -> begin
        hi = max(abs(vmin), abs(vmax))
        emax = max(hi > 0. ? floor(Int, log10(hi)) : e0, e0)
        estep = max(1, cld(2 * (emax - e0 + 1) + 1, max_ticks))
        vals = [0.]
        for e in e0:estep:emax
            push!(vals, 10.0^e, -10.0^e)
        end
        sort!(vals)
        filter!(v -> (vmin <= v <= vmax) && ((v == 0.) || (abs(v) > trim * linthresh)), vals)
        vals, ticklabel.(vals)
    end
end

"""
    MinorTicks(n=9)

`n` minor ticks between every pair of neighbouring major ticks, evenly spaced in
data space - so the step is constant within each major tick interval, and how much
they bunch up in the plot shows how far the scale distorts data space there. Use as
`yminorticks=CustomSymlogScale.MinorTicks(4)` along with `yminorticksvisible=true`,
or via `kwargs`. Takes the major ticks as Makie computes them, so it follows
`max_ticks` and `trim` - the wider apart those leave the majors, the more extreme
the bunching.
"""
struct MinorTicks
    n::Int
end
MinorTicks() = MinorTicks(9)
function Makie.get_minor_tickvalues(mt::MinorTicks, _, tickvalues, vmin, vmax)
    vs = Float64[]
    for (lo, hi) in zip(tickvalues[1:(end-1)], tickvalues[2:end])
        append!(vs, range(lo, hi, mt.n + 2)[2:(end-1)])
    end
    filter!(v -> vmin <= v <= vmax, vs)
end

"""
    guess_linthresh(vals...; floor_frac=1e-6)

Pick a linthresh for `vals`: the nearest decade to the smallest nonzero magnitude,
but never more than `floor_frac` below the largest one.
"""
function guess_linthresh(vals...; floor_frac=1e-6)
    as = filter(>(0.), abs.(reduce(vcat, map(collect, vals))))
    isempty(as) && return 1.
    10.0^round(Int, log10(max(minimum(as), floor_frac * maximum(as))))
end

"""
    kwargs(linthresh; dim=:y, kwargs...)

All the `Axis` kwargs needed for a symlog `dim` axis at once, e.g.
`Axis(fig[1,1]; CustomSymlogScale.kwargs(1e-6)...)`.
"""
function kwargs(linthresh; dim=:y, max_ticks=11, trim=0., nminor=9, minorgrid=false)
    kws = (;
        scale=scale(linthresh),
        ticks=ticks(linthresh; max_ticks, trim),
        minorticks=MinorTicks(nminor),
        minorticksvisible=true,
        minorgridvisible=minorgrid,
    )
    NamedTuple(Symbol(dim, k) => v for (k, v) in pairs(kws)) # prefix each key with dim
end

end
export CustomSymlogScale

################################################################################
# Sub/optional bits that come in submodules
################################################################################
# Minimal model specific bits
include("MinimalModelSemisymbolic/MinimalModelSemisymbolic.jl")
include("MinimalModelSemisymbolic/MinimalModelV2.jl")
include("MinimalModelSemisymbolic/MinimalModelV3.jl")
include("SymCosmo/SymCosmo.jl")
include("RandomSystems/RandomSystems.jl")
include("TwoMMs.jl")

include("FFTAnalysis.jl")

end
