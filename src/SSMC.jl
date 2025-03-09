module SSMC

using REPL.TerminalMenus
using Logging

using StaticArrays
using Makie

import Base: show

projectdir(args...) = joinpath(pkgdir(LuriaDelbruck), args...)
scriptsdir(args...) = joinpath(projectdir(), "scripts", args...)
datadir(args...) = joinpath(projectdir(), "data", args...)
export projectdir, scriptsdir, datadir

################################################################################
# Utility bits
################################################################################
function uninplace(f)
    function (u, args...)
        du = similar(u)
        f(du, u, args...)
        du
    end
end
export uninplace

function smart_val(passed_val, default, shape...)
    val = isnothing(passed_val) ? default : passed_val
    if isnothing(val)
        throw(ArgumentError("encountered a nothing argument which does not have a default"))
    end
    if !isa(val, AbstractArray)
        val = fill(val, shape)
    end
    val
end
smart_sval(pv, d, shape...) = SArray{Tuple{shape...}}(smart_val(pv, d, shape...))
split_name_args(x) = x, ()
split_name_args(t::Tuple) = t[1], t[2:end]
export smart_val, smart_sval, split_name_args

function prep_paramscan(; param_ranges...)
    ranges_only = []
    df_colspecs = Pair{Symbol,Any}[]
    for (pname, prange) in param_ranges
        if !isa(prange, AbstractVector)
            prange = [prange]
        end
        push!(ranges_only, prange)
        push!(df_colspecs, pname => empty(prange))
    end

    cis = CartesianIndices(Tuple(length.(ranges_only)))
    itoparams = function (i)
        [r[i] for (r, i) in zip(ranges_only, cis[i].I)]
    end

    df_colspecs, itoparams, cis
end
export prep_paramscan

# FIX: Should be removed as prep_paramscan is a better version of it
function setup_ranges(ranges...; func=identity)
    cis = CartesianIndices(length.(ranges))
    function (i)
        func([r[i] for (r, i) in zip(ranges, cis[i].I)])
    end, cis
end
export setup_ranges

function wait_till_confirm()
    menu = RadioMenu(["continue", "maybe exit"], pagesize=2)
    request(menu) == 1
end
export wait_till_confirm

# These are needed for Jupyter...
function fast_info(msg)
    @info msg
    flush(Logging.current_logger().stream)
end
function fast_warn(msg)
    @warn msg
    flush(Logging.current_logger().stream)
end
export fast_info, fast_warn

struct FigureAxisAnything
    figure::Figure
    axis::Any
    obj::Any
end
show(io::IO, faa::FigureAxisAnything) = show(io, faa.figure)
show(io::IO, mime, faa::FigureAxisAnything) = show(io, mime, faa.figure)
export FigureAxisAnything

################################################################################
# Other modules/files
################################################################################
include("MLSolver.jl")
include("BasicMiCRM.jl")

end # module SSMC
