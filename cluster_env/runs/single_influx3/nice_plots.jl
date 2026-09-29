"""
nice_plots.jl

Standalone plotting for this run directory's `do_Kli_run` output (see
`base.jl`) — any `main*.jld2` file here holding a `df` with
`Kis`/`liis`/`codes` columns and a `metadata` with `Ks`/`lis`.

Outcome codes, per `do_Kli_run`:
  1  => extinct
  2  => stable (well-mixed AND spatially stable)
  3  => spatially unstable (well-mixed stable, unstable to some wavenumber)
 -1  => solver did not return Success
 -2  => well-mixed unstable
 -3  => residual too high (not converged)
`-1`/`-2`/`-3` are lumped together below as "Bad".

For each data file, produces (by default into its own `<base>_plots`
directory, `<base>` = that file's name without extension):
  - `<base>_outcomes.pdf`     ternary-coloured Extinct/Stable/Unstable scatter
                              over the (K, leakage) grid, with the analytic
                              beta_v/beta_s boundary curves overlaid
  - `<base>_proportions.pdf`  grid of Extinct/Stable/Unstable/Bad proportion-vs-K
                              panels, one panel per leakage value
  - `<base>_unstable.pdf`     Unstable proportion vs K (with Clopper-Pearson
                              confidence bands), one line per leakage value,
                              all on one axis, colour-coded by leakage with a
                              matching Colorbar

Use from the command line, with one or more data files at once (the
plotting packages are only loaded once, not per file):

    julia --project nice_plots.jl main2_B5.jld2 main6_B10.jld2 main7_B20.jld2

By default each file gets its own `<base>_plots` directory; pass `-o`/
`--outdir` to put every file's plots into one shared directory instead:

    julia --project nice_plots.jl -o all_plots main2_B5.jld2 main6_B10.jld2

or `include` this file into a notebook and call `outcome_plot`,
`proportions_plot`, or `nice_plots` (single file or a vector of files)
directly — including it does not run anything or touch the active Makie
backend by itself; each plotting function switches to CairoMakie itself
right before drawing (reactivate GLMakie afterwards if you need it for
interactive work).
"""

using JLD2
using DataFrames
using CairoMakie
using Printf
using ArgParse
using HypothesisTests
using SSMCMain, SSMCMain.ModifiedMiCRM
import SSMCMain.ModifiedMiCRM.MinimalModelV3

const EXTINCT_CODE = 1
const STABLE_CODE = 2
const UNSTABLE_CODE = 3
const BAD_CODES = (-1, -2, -3)

# Plain colorant"..." strings and RGBf are re-exported by CairoMakie itself, so
# these don't need Colors as a direct dependency — deliberately not reusing
# scripts/ternary_colormap.jl here since its explicit `using Colors` fails
# under cluster_env/Project.toml (Colors is only a transitive dependency
# there), which is the environment these plots actually get made under.
const extinct_color = colorant"#898989"
const stable_color = colorant"#1b9e77"
const unstable_color = colorant"#d95f02"
const bad_color = colorant"#000000"

"""
    ternary_blend(e, s, u)

Simple linear sRGB blend of the extinct/stable/unstable corner colours,
weighted by the (non-negative) counts/fractions `e`, `s`, `u`.
"""
function ternary_blend(e, s, u)
    total = e + s + u
    total > 0 || throw(ArgumentError("e, s, u must not all be zero"))
    w = (e, s, u) ./ total
    RGBf((w[1] .* Tuple(extinct_color) .+ w[2] .* Tuple(stable_color) .+ w[3] .* Tuple(unstable_color))...)
end

"""
    load_run(fname) -> (df, metadata)

Load a `do_Kli_run`-style jld2 file, closing the file handle again afterwards.
"""
function load_run(fname)
    jldopen(fname, "r") do f
        (f["df"], f["metadata"])
    end
end

"""
    outcome_count_matrices(df) -> Dict{Int,Matrix{Int}}

Count outcome codes per (K index, leakage index) grid cell. Each matrix is
`numKs x numlis`, keyed by code.
"""
function outcome_count_matrices(df)
    numKs = maximum(df.Kis)
    numlis = maximum(df.liis)
    mats = Dict{Int,Matrix{Int}}(code => zeros(Int, numKs, numlis) for code in unique(df.codes))
    for r in eachrow(df)
        mats[r.codes][r.Kis, r.liis] += 1
    end
    mats
end

"""
    default_outdir(fname) -> String

`<base>_plots`, where `<base>` is `fname`'s basename without extension.
"""
default_outdir(fname) = first(splitext(basename(fname))) * "_plots"

"""
    load_outcome_matrices(fname) -> (; Ks, lis, extinct, stable, unstable, bad, total)

Load `fname` and return its K/leakage axes together with the Extinct/Stable/
Unstable/Bad outcome-code count matrices (each `numKs x numlis`) and their
elementwise sum `total`.
"""
function load_outcome_matrices(fname)
    df, metadata = load_run(fname)
    Ks = metadata.Ks
    lis = metadata.lis
    zeromat = zeros(Int, length(Ks), length(lis))
    mats = outcome_count_matrices(df)
    extinct = get(mats, EXTINCT_CODE, zeromat)
    stable = get(mats, STABLE_CODE, zeromat)
    unstable = get(mats, UNSTABLE_CODE, zeromat)
    bad = sum(get(mats, c, zeromat) for c in BAD_CODES)
    (; Ks, lis, extinct, stable, unstable, bad, total=extinct .+ stable .+ unstable .+ bad)
end

"""
    outcome_plot(fname; outdir=nothing, outname=nothing, gpcols=[(1., 1., :black)])

Ternary-coloured Extinct/Stable/Unstable scatter of the (K, leakage) grid in
`fname`, with the analytic viability (`beta_v`) and qualified-instability
(`beta_s`) curves overlaid for each `(gamma, p, color)` in `gpcols`. Grid cells
with no Extinct/Stable/Unstable repeats at all (i.e. every repeat was "Bad")
are drawn in `bad_color` rather than erroring.

Saves to `<outdir>/<base>_outcomes.pdf` (default `outdir` from
[`default_outdir`](@ref)) and returns the `Figure`.
"""
function outcome_plot(fname;
    outdir=nothing,
    outname=nothing,
    gpcols=[(1., 1., :black)],
)
    CairoMakie.activate!()

    base = first(splitext(basename(fname)))
    outdir = something(outdir, default_outdir(fname))
    mkpath(outdir)
    outpath = joinpath(outdir, something(outname, base * "_outcomes.pdf"))

    m = load_outcome_matrices(fname)
    Ks = m.Ks
    leakxs = LeakageScale.ltox.(m.lis)

    safe_blend(e, s, u) = (e + s + u) > 0 ? ternary_blend(e, s, u) : bad_color
    colors = safe_blend.(vec(m.extinct'), vec(m.stable'), vec(m.unstable'))

    fig = Figure()
    ax = Axis(fig[1, 1];
        xscale=log10,
        title=(@sprintf "fname=%s" fname),
        xlabel="Normalized energy supply rate",
        ylabel="Supplied resource leakage",
        xgridvisible=false,
        ygridvisible=false,
    )
    eps_ticks = [0.5, 0.3, 0.1, 0.01, 0.001]
    ax.yticks = (LeakageScale.etox.(eps_ticks), [(@sprintf "%.3g" (1 - e)) for e in eps_ticks])
    ax.yminorticks = LeakageScale.exminorticks(eps_ticks, 4)

    scatter!(ax, [(x, y) for x in Ks for y in leakxs]; markersize=20, color=colors)

    for (gamma, p, color) in gpcols
        ls2 = LeakageScale.l.(range(extrema(leakxs)..., 1000))
        extline_Ks = MinimalModelV3.fr3_beta_viable.(ls2, gamma)
        instabline_Ks = MinimalModelV3.fr3_beta_s_qualified.(ls2, gamma, p)
        leakxs2 = LeakageScale.ltox.(ls2)

        lines!(ax, extline_Ks, leakxs2; color, label=(@sprintf "beta_v with gamma=%.3g, p=%.3g" gamma p))
        lines!(ax, instabline_Ks, leakxs2; color, linestyle=:dash, label=(@sprintf "beta_s with gamma=%.3g, p=%.3g" gamma p))
    end
    axislegend(ax; position=:rb)

    CairoMakie.save(outpath, fig)
    fig
end

"""
    proportions_plot(fname; outdir=nothing, outname=nothing, ncols=nothing)

Grid of Extinct/Stable/Unstable/Bad proportion-vs-K panels, one panel per
leakage value in `fname`'s grid. `ncols` defaults to `min(numlis, 5)`.

Saves to `<outdir>/<base>_proportions.pdf` and returns the `Figure`.
"""
function proportions_plot(fname;
    outdir=nothing,
    outname=nothing,
    ncols=nothing,
)
    CairoMakie.activate!()

    base = first(splitext(basename(fname)))
    outdir = something(outdir, default_outdir(fname))
    mkpath(outdir)
    outpath = joinpath(outdir, something(outname, base * "_proportions.pdf"))

    m = load_outcome_matrices(fname)
    Ks = m.Ks
    lis = m.lis
    numlis = length(lis)
    safe_total = replace(m.total, 0 => 1) # avoid 0/0 for grid cells with no repeats at all

    ncols = something(ncols, min(numlis, 5))
    nrows = cld(numlis, ncols)

    fig = Figure(size=(280 * ncols, 220 * nrows + 60))
    series = [
        ("Extinct", m.extinct, extinct_color),
        ("Stable", m.stable, stable_color),
        ("Unstable", m.unstable, unstable_color),
        ("Bad", m.bad, bad_color),
    ]
    plots = nothing
    for (li_i, li) in enumerate(lis)
        row = div(li_i - 1, ncols) + 1
        col = mod(li_i - 1, ncols) + 1
        ax = Axis(fig[row, col];
            xscale=log10,
            title=(@sprintf "li=%.4g" li),
            xlabel="K",
            ylabel="Proportion",
        )
        plots = [
            scatterlines!(ax, Ks, mat[:, li_i] ./ safe_total[:, li_i]; color, markersize=6, label=name)
            for (name, mat, color) in series
        ]
    end
    Legend(fig[nrows+1, 1:ncols], plots, first.(series); orientation=:horizontal, tellwidth=false)

    CairoMakie.save(outpath, fig)
    fig
end

"""
    unstable_plot(fname; outdir=nothing, outname=nothing, cmap=:viridis, ci_level=0.95)

Unstable-outcome (code 3) proportion vs K, one line per leakage value in
`fname`'s grid, all on a single axis instead of `proportions_plot`'s
one-panel-per-leakage grid. Lines are colour-coded by
`LeakageScale.ltox(li)` — the same transform used for the y-axis in
[`outcome_plot`](@ref) — with a matching `Colorbar`. Each line carries a
shaded `ci_level` Clopper-Pearson confidence band (`HypothesisTests.jl`'s
default `BinomialTest` interval).

Saves to `<outdir>/<base>_unstable.pdf` and returns the `Figure`.
"""
function unstable_plot(fname;
    outdir=nothing,
    outname=nothing,
    cmap=:viridis,
    ci_level=0.95,
)
    CairoMakie.activate!()

    base = first(splitext(basename(fname)))
    outdir = something(outdir, default_outdir(fname))
    mkpath(outdir)
    outpath = joinpath(outdir, something(outname, base * "_unstable.pdf"))

    m = load_outcome_matrices(fname)
    Ks = m.Ks
    leakxs = LeakageScale.ltox.(m.lis)
    crange = extrema(leakxs)

    fig = Figure()
    ax = Axis(fig[1, 1];
        xscale=log10,
        title=(@sprintf "fname=%s" fname),
        xlabel="Normalized energy supply rate",
        ylabel="Unstable proportion",
    )

    for li_i in eachindex(leakxs)
        n = @view m.total[:, li_i]
        k = @view m.unstable[:, li_i]
        p = k ./ replace(n, 0 => 1)
        los = similar(p)
        his = similar(p)
        for i in eachindex(n)
            los[i], his[i] = n[i] > 0 ? confint(BinomialTest(k[i], n[i]); level=ci_level) : (0.0, 0.0)
        end
        color = leakxs[li_i]
        band!(ax, Ks, los, his; color, colorrange=crange, colormap=cmap, alpha=0.25)
        lines!(ax, Ks, p; color, colorrange=crange, colormap=cmap)
    end

    eps_ticks = [0.5, 0.3, 0.1, 0.01, 0.001]
    Colorbar(fig[1, 2];
        limits=crange,
        colormap=cmap,
        label="Supplied resource leakage",
        ticks=(LeakageScale.etox.(eps_ticks), [(@sprintf "%.3g" (1 - e)) for e in eps_ticks]),
    )

    CairoMakie.save(outpath, fig)
    fig
end

"""
    nice_plots(fname; outdir=nothing, gpcols=[(1., 1., :black)], ncols=nothing, ci_level=0.95)

Convenience wrapper producing [`outcome_plot`](@ref), [`proportions_plot`](@ref)
and [`unstable_plot`](@ref) for `fname` into the same output directory. Returns
`(outcome=fig1, proportions=fig2, unstable=fig3)`.
"""
function nice_plots(fname; outdir=nothing, gpcols=[(1., 1., :black)], ncols=nothing, ci_level=0.95)
    outdir = something(outdir, default_outdir(fname))
    (
        outcome=outcome_plot(fname; outdir, gpcols),
        proportions=proportions_plot(fname; outdir, ncols),
        unstable=unstable_plot(fname; outdir, ci_level),
    )
end

"""
    nice_plots(fnames::AbstractVector; outdir=nothing, gpcols=[(1., 1., :black)], ncols=nothing, ci_level=0.95)

Same as the single-file method, run over several files (loading the plotting
packages only once). `outdir` — if given — is shared by every file; otherwise
each file gets its own `<base>_plots`. Returns a `Dict` from `fname` to its
`(outcome=.., proportions=.., unstable=..)` result.
"""
function nice_plots(fnames::AbstractVector; outdir=nothing, gpcols=[(1., 1., :black)], ncols=nothing, ci_level=0.95)
    Dict(fname => nice_plots(fname; outdir, gpcols, ncols, ci_level) for fname in fnames)
end

function parse_cli_args(args)
    s = ArgParseSettings(;
        description="Produce the outcome-scatter and proportion plots for one or more do_Kli_run data files in this directory (e.g. main2_B5.jld2).",
    )
    @add_arg_table! s begin
        "datafiles"
            help = "one or more main*.jld2 files produced by do_Kli_run"
            nargs = '+'
            required = true
        "--outdir", "-o"
            help = "put every file's plots here instead of each getting its own <base>_plots"
            default = nothing
    end
    parse_args(args, s)
end

function main(args)
    parsed = parse_cli_args(args)
    outdir = parsed["outdir"]
    for fname in parsed["datafiles"]
        nice_plots(fname; outdir)
        println("wrote plots for $fname to $(something(outdir, default_outdir(fname)))/")
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
