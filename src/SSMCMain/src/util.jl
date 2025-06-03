################################################################################
# Handling arguments and similar bits
################################################################################
function smart_val(passed_val, default, target_size...)
    val = isnothing(passed_val) ? default : passed_val
    if isnothing(val)
        throw(ArgumentError("encountered a nothing argument which does not have a default"))
    end
    if !isa(val, AbstractArray)
        val = fill(val, target_size)
    end
    if size(val) != target_size
        throw(ArgumentError(@sprintf "could not correctly size passed argument, got %s and want %s" string(size(val)) string(target_size)))
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

################################################################################
# Parameter scanning
################################################################################
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

function prep_paramscan_floats(; param_ranges...)
    # Pre-allocate with known size
    n_params = length(param_ranges)
    ranges_only = Vector{Vector{Float64}}(undef, n_params)
    df_colspecs = Vector{Pair{Symbol,Any}}(undef, n_params)

    # Process parameters in a single loop
    for (i, (pname, prange)) in enumerate(param_ranges)
        ranges_only[i] = isa(prange, AbstractVector) ? prange : [prange]
        if eltype(ranges_only[i]) != Float64
            @warn "converting some non-float parameters to float"
        end
        df_colspecs[i] = pname => empty(ranges_only[i])
    end

    cis = CartesianIndices((length.(ranges_only)...,))

    # Use closure to capture ranges_only
    itoparams = i -> getindex.(ranges_only, cis[i].I)

    df_colspecs, itoparams, cis
end
export prep_paramscan_floats

################################################################################
# Miscelanious
################################################################################
function uninplace(f)
    function (u, args...)
        du = similar(u)
        f(du, u, args...)
        du
    end
end
export uninplace

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

function extend_solprob(sol, T)
    prob = sol.prob

    tspan = (sol.t[end], sol.t[end] + T)

    remake(prob, u0=sol.u[end], tspan=tspan)
end
export extend_solprob

function remake_guarantee_positive(prob)
    fclosure = let zz = zero(eltype(prob.u0))
        (u, _, _) -> minimum(u) < zz
    end
    remake(prob; isoutofdomain=fclosure)
end
export remake_guarantee_positive

function linear_spacing_edges(xs)
    ss = similar(xs, length(xs) + 1)
    for i in 1:(length(xs)-1)
        ss[i+1] = (xs[i] + xs[i+1]) / 2
    end
    ss[1] = xs[1] - (ss[2] - xs[1])
    ss[end] = xs[end] + (xs[end] - ss[end-1])
    ss
end
export linear_spacing_edges

################################################################################
# Plotting
################################################################################
struct FigureAxisAnything
    figure::Figure
    axis::Any
    obj::Any
end
show(io::IO, faa::FigureAxisAnything) = show(io, faa.figure)
show(io::IO, mime, faa::FigureAxisAnything) = show(io, mime, faa.figure)
export FigureAxisAnything

function make_grid(n;
    aspect_ratio=1.0, prioritize=:none,
    min_rows=1, max_rows=typemax(Int),
    min_cols=1, max_cols=typemax(Int)
)
    # Calculate initial number of columns based on aspect ratio
    cols = ceil(Int, sqrt(n * aspect_ratio))
    rows = ceil(Int, n / cols)

    # Adjust based on prioritization
    if prioritize == :rows
        rows = min(max(min_rows, rows), max_rows)
        cols = ceil(Int, n / rows)
    elseif prioritize == :cols
        cols = min(max(min_cols, cols), max_cols)
        rows = ceil(Int, n / cols)
    end

    # Ensure the grid fits within the specified limits
    rows = min(max(min_rows, rows), max_rows)
    cols = min(max(min_cols, cols), max_cols)

    return rows, cols
end
export make_grid

################################################################################
# Plotting DimensionalData.jl stuff
################################################################################
function plot_2ddimdata_heatmap_int(data, xvar, yvar;
    colorbar=true,
    kwargs...
)
    fig = Figure()

    odims = otherdims(data, xvar, yvar)
    odim_names = name.(odims)

    specs = [(label=string(name(od)), range=val(od)) for od in odims]
    sg = SliderGrid(fig[1, 1], specs...)

    sovals = [s.value for s in sg.sliders]
    marginalized_data = lift(sovals...) do svals...
        getindex(data; (odim_names .=> At.(svals))...)
    end

    ax = Axis(fig[2, 1])
    ax.xlabel = string(xvar)
    ax.ylabel = string(yvar)
    hm = heatmap!(ax, marginalized_data; kwargs...)

    if colorbar
        cb = Colorbar(fig[2, 2], hm)
    end

    FigureAxisAnything(fig, ax, marginalized_data)
end
export plot_2ddimdata_heatmap_int

################################################################################
# Plotting NamedArrays with labels etc plus DataFrames helpers
################################################################################
function plot_namedvector_numeric!(ax, nv::NamedVector;
    logx=nothing,
    kwargs...
)
    xs = names(nv)[1]
    xlabel = string(dimnames(nv)[1])
    if logx == true
        logx = 10
    end
    if !isnothing(logx)
        xs = log.(logx, xs)
        xlabel = "log$logx($xlabel)"
    end

    p = scatter!(ax, xs, nv; kwargs...)
    ax.xlabel = xlabel

    p
end
function plot_namedvector_numeric(args...; kwargs...)
    fig = Figure()
    ax = Axis(fig[1, 1])
    p = plot_namedvector_numeric!(ax, args...; kwargs...)
    FigureAxisAnything(fig, ax, p)
end
export plot_namedvector_numeric!, plot_namedvector_numeric

function plot_namedmatrix_numeric!(ax, nm::NamedMatrix;
    logx=nothing, logy=nothing,
    colorbarplace=nothing, kwargs...
)
    xs, ys = names(nm)
    xlabel, ylabel = string.(dimnames(nm))
    if logx == true
        logx = 10
    end
    if !isnothing(logx)
        xs = log.(logx, xs)
        xlabel = "log$logx($xlabel)"
    end
    if logy == true
        logy = 10
    end
    if !isnothing(logy)
        ys = log.(logy, ys)
        ylabel = "log$logy($ylabel)"
    end

    p = heatmap!(ax, xs, ys, nm; kwargs...)
    ax.xlabel = xlabel
    ax.ylabel = ylabel

    if !isnothing(colorbarplace)
        Colorbar(colorbarplace, p)
    end

    p
end
function plot_namedmatrix_numeric(args...; colorbar=true, kwargs...)
    fig = Figure()
    ax = Axis(fig[1, 1])
    if colorbar
        p = plot_namedmatrix_numeric!(ax, args...; colorbarplace=fig[1, 2], kwargs...)
    else
        p = plot_namedmatrix_numeric!(ax, args...; kwargs...)
    end
    FigureAxisAnything(fig, ax, p)
end
export plot_namedmatrix_numeric!, plot_namedmatrix_numeric

function plot_namedmatrix_discrete!(ax, nm::NamedMatrix;
    colorbarplace=nothing, kwargs...
)
    p = heatmap!(ax, nm; kwargs...)
    xns, yns = names(nm)
    ax.xticks = (1:length(xns), string.(xns))
    ax.yticks = (1:length(yns), string.(yns))

    if !isnothing(colorbarplace)
        Colorbar(colorbarplace, p)
    end

    p
end
function plot_namedmatrix_discrete(args...; colorbar=true, kwargs...)
    fig = Figure()
    ax = Axis(fig[1, 1])
    if colorbar
        p = plot_namedmatrix_discrete!(ax, args...; colorbarplace=fig[1, 2], kwargs...)
    else
        p = plot_namedmatrix_discrete!(ax, args...; kwargs...)
    end
    FigureAxisAnything(fig, ax, p)
end
export plot_namedmatrix_discrete!, plot_namedmatrix_discrete

function plot_df_correlations(df, select=:;
    clear_diag=true,
    full_range=false,
    kwargs...
)
    df = df[!, select]
    corrs = NamedArray(cor(Matrix(df)), (names(df), names(df)))

    if clear_diag
        for i in 1:size(corrs, 1)
            corrs[i, i] = NaN
        end
    end

    colorrange = if full_range
        (-1.0, 1.0)
    else
        mm = maximum(abs, filter(isfinite, corrs))
        (-mm, mm)
    end

    fap = plot_namedmatrix_discrete(corrs;
        colorrange, colormap=:RdBu,
        kwargs...
    )
    if isempty(fap.axis.title[])
        fap.axis.title = "Correlations"
    end
    if (maximum(length, fap.axis.xticks[][2]) > 6) && iszero(fap.axis.xticklabelrotation[])
        fap.axis.xticklabelrotation = 0.4
    end

    fap
end
export plot_df_correlations

################################################################################
# Other miscellaneous plotting
################################################################################
function plot_heatmaps(xs, ys, matrices;
    titles=nothing
)
    n = length(matrices)
    nrows, ncols = make_grid(length(matrices))

    fig = Figure()
    axes = []
    hms = []

    for i in 1:n
        row = div(i - 1, ncols) + 1
        col = (mod(i - 1, ncols) + 1) * 2 - 1
        ax = Axis(fig[row, col])
        push!(axes, ax)

        if isnothing(xs) && isnothing(ys)
            hm = heatmap!(ax, matrices[i])
        else
            hm = heatmap!(ax, xs, ys, matrices[i])
        end
        push!(hms, hm)
        Colorbar(fig[row, col+1], hm)

        if !isnothing(titles) && !isnothing(titles[i])
            title!(ax, titles[i])
        end
    end

    FigureAxisAnything(fig, axes, hms)
end
plot_heatmaps(matrices; kwargs...) = plot_heatmaps(nothing, nothing, matrices; kwargs...)
export plot_heatmaps

################################################################################
# Distribution demos
################################################################################
function dist_demo_normal(xs=range(-10.0, 10.0, 1000))
    fig = Figure()
    sg = SliderGrid(fig[1, 1],
        (; label="μ", range=-10.0:0.0001:10.0, startvalue=0.0),
        (; label="σ", range=0.0:0.0001:10.0, startvalue=1.0),
    )
    mu_o = sg.sliders[1].value
    sigma_o = sg.sliders[2].value

    pdf_vals = lift(mu_o, sigma_o) do mu, sigma
        pdf(Normal(mu, sigma), xs)
    end

    ax = Axis(fig[2, 1])
    lines!(ax, xs, pdf_vals)

    fig
end
export dist_demo_normal

function dist_demo_lognormal(xs=range(0.0, 100.0, 1000))
    fig = Figure()
    sg = SliderGrid(fig[1, 1],
        (; label="μ", range=0.0:0.0001:10.0, startvalue=1.0),
        (; label="σ", range=0.0:0.0001:10.0, startvalue=1.0),
    )
    mu_o = sg.sliders[1].value
    sigma_o = sg.sliders[2].value

    pdf_vals = lift(mu_o, sigma_o) do mu, sigma
        pdf(LogNormal(mu, sigma), xs)
    end

    ax = Axis(fig[2, 1])
    lines!(ax, xs, pdf_vals)

    fig
end
export dist_demo_lognormal

################################################################################
# OLD
################################################################################
# FIX: Should be removed as prep_paramscan is a better version of it
function setup_ranges(ranges...; func=identity)
    cis = CartesianIndices(length.(ranges))
    function (i)
        func([r[i] for (r, i) in zip(ranges, cis[i].I)])
    end, cis
end
export setup_ranges
