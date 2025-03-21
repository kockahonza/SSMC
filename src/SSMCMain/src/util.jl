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

function smart_val(passed_val, default, target_size...)
    val = isnothing(passed_val) ? default : passed_val
    if isnothing(val)
        throw(ArgumentError("encountered a nothing argument which does not have a default"))
    end
    if !isa(val, AbstractArray)
        val = fill(val, target_size)
    end
    if size(val) != target_size
        throw(ArgumentError("could not correctly size passed argument"))
    end
    val
end
smart_sval(pv, d, target_size...) = SArray{Tuple{target_size...}}(smart_val(pv, d, target_size...))
split_name_args(x) = x, ()
split_name_args(t::Tuple) = t[1], t[2:end]
export smart_val, smart_sval, split_name_args

function splitkwargs(kwargs, args...)
    out = []
    unused_keys = keys(kwargs)
    for kwarg in args
        some_keys = intersect(keys(kwargs), kwarg)
        unused_keys = filter(k -> !(k in some_keys), unused_keys)
        push!(out, (; [(key, kwargs[key]) for key in some_keys]...))
    end
    push!(out, (; [(key, kwargs[key]) for key in unused_keys]...))
    out
end
export splitkwargs

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
# Plotting
################################################################################
function plot_linstab_lambdas(ks, lambdas; imthreshold=1e-8)
    fig = Figure()
    ax = Axis(fig[1, 1])
    for li in axes(lambdas, 2)
        lines!(ax, ks, real(lambdas[:, li]);
            color=Cycled(li),
            label=latexstring(@sprintf "\\Re(\\lambda_%d)" li)
        )
        ims = imag(lambdas[:, li])
        mims = maximum(abs, ims)
        if mims > imthreshold
            @info @sprintf "we are getting non-zero imaginary parts, max(abs(.)) is %f" mims
            lines!(ax, ks, ims;
                color=Cycled(li),
                linestyle=:dash,
                label=latexstring(@sprintf "\\Im(\\lambda_%d)" li)
            )
        end
    end
    axislegend(ax)
    FigureAxisAnything(fig, ax, lambdas)
end
export plot_linstab_lambdas
