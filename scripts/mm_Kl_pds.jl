using SSMCMain, SSMCMain.ModifiedMiCRM.MinimalModelV2

using Revise

using OhMyThreads
using ProgressMeter
using JLD2
using ColorSchemes

includet(scriptsdir("figures_util.jl"))
using Makie

function do_Kl_pd_run(
    logKs, ls,
    m, c,
    DN, DI, DR,
    d, k, r
    ;
    return_raw=false,
)
    rslts = Matrix{Vector{NospaceSolStability.T}}(undef, length(logKs), length(ls))
    @tasks for i in 1:length(logKs)
        logK = logKs[i]
        for (j, l) in enumerate(ls)
            rslts[i, j] = analyse_mmp(
                MMParams(;
                    K=10^logK,
                    m,
                    l,
                    k,
                    c,
                    d,
                    r,
                );
                DN, DI, DR,
            )
        end
    end
    if return_raw
        rslts
    else
        nospacesolstabilities_to_code.(rslts)
    end
end

function make_Kl_pd!(
    place,
    logKs, ls,
    m, c,
    DN, DI, DR;
    r=1.0
)
    leak_xs = LeakageScale.ltox.(ls)

    xx = do_Kl_pd_run(
        logKs, ls,
        m, c,
        DN, DI, DR,
        c, 0.0, r
    )
    cs = PaperColors.mma_coloring_full.(xx)

    ax = MinimalModelV2.make_mm_Kl_hm_ax(place[1, 1], logKs, ls)

    hm = heatmap!(ax, 10 .^ logKs, leak_xs, cs;
        rasterize=3
    )

    MinimalModelV2.draw_fr_lines2!(ax, ls, m, c, DR / DI)

    ax, hm
end
function make_Kl_pd(args...)
    fig = Figure(;
        # size=(double_col_width * 0.25, double_col_width * 0.18),
        # default_fig_kwargs...,
        # # figure_padding=tuple(fill(3., 4))
        # figure_padding=(2.0, 8.0, 2.0, 2.0)
    )

    ax, hm = make_Kl_pd!(fig[1, 1], args...)

    Makie.FigureAxisPlot(fig, ax, hm)
end

function make_Kl_pd_legend_full!(place; kwargs...)
    Legend(place,
        [MarkerElement(; color=c, marker=:rect) for c in [
            PaperColors.extinct1(),
            PaperColors.extinct2(),
            PaperColors.extinct3(),
            PaperColors.stable1(),
            PaperColors.stable2(),
            PaperColors.unstable1(),
            PaperColors.unstable2(),
            PaperColors.other(),
        ]],
        [
            "Extinct 1 - no positive real sols",
            "Extinct 2 - one posreal sol, nospace unstable",
            "Extinct 3 - two posreal sols, both nospace unstable",
            "Stable 1 - one posreal sol, spatially stable",
            "Stable 2 - two posreal sols, one nospace unstable and one spatially stable",
            "Unstable 1 - one posreal sol, spatially unstable",
            "Unstable 2 - two posreal sols, one nospace unstable and one spatially unstable",
            "Other - anything else",
        ];
        kwargs...
    )
end
