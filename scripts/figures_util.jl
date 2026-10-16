using Revise
includet("./ternary_colormap.jl")

# figure sizes for publication
single_col_width = 324 * 1.051437737 # corresponds to 511 pdf pt which is ~9cm
double_col_width = single_col_width * 2
golden_ratio = 1.618
pt_per_mm_ratio = 3.779527559

fontsize_pt = (4/3)
fontsize_ltex_pt = (4/3)*(9/8)

fig_fontsize = 8

default_fig_kwargs = (;
    fontsize=6fontsize_pt,
    markersize=6,
    figure_padding=(2, 2, 2, 2)
)

# colors to use everywhere
module PaperColors
using ColorSchemes
using Makie
using ..TernaryColormaps

# Setting colours for the different outcomes
extinct1() = ColorSchemes.Blues[5]
extinct2() = ColorSchemes.Blues[7]
extinct3() = ColorSchemes.Blues[9]
stable1() = ColorSchemes.Oranges[4]
stable2() = ColorSchemes.Oranges[5]
unstable1() = ColorSchemes.Greens[4]
unstable2() = ColorSchemes.Greens[5]
other() = ColorSchemes.Dark2_4[end]

extinct() = colorant"#898989"
stable() = colorant"#1b9e77"
unstable() = colorant"#d95f02"

outcome_ternary_colormap() = TernaryColormap(PaperColors.extinct(), PaperColors.stable(), PaperColors.unstable())

function mma_coloring_full(code)
    map_ = Dict(
        0 => extinct1(),
        1 => extinct2(),
        11 => extinct3(),
        2 => stable1(),
        12 => stable2(),
        3 => unstable1(),
        13 => unstable2(),
    )

    get(map_, code, other())
end
function mma_coloring_simple(code)
    map_ = Dict(
        0 => extinct1(),
        1 => extinct1(),
        11 => extinct1(),
        2 => stable1(),
        12 => stable1(),
        3 => unstable1(),
        13 => unstable1(),
    )

    get(map_, code, other())
end

mm_extline() = ColorSchemes.Dark2_6[end-4]
mm_instabline() = ColorSchemes.Dark2_6[end-3]

twomms_1() = ColorSchemes.Reds[end-2]
twomms_2() = ColorSchemes.Blues[end-2]
twomms_coexistence() = colorant"#b53296"

# Defining a single p colormap to use throughout
p_colormap = ColorScheme(ColorSchemes.Blues[5:end])
p_range = (1e-1, 2)
p_logrange = log10.(p_range)

p_color1(p) = get(p_colormap, log10(p), p_logrange)

# for a Colorbar labelled in p rather than log10(p)
p_colorbar1_kwargs(ps) = (;
    colormap=p_colormap,
    colorrange=p_logrange,
    ticks=(log10.(ps), [rich("10", superscript(string(Int(log10(p))))) for p in ps]),
)
function p_colorbar1(ps=10.0 .^ (-2:1))
    fig = Figure(; size=(200, 50))
    Makie.Colorbar(fig[1, 1];
        vertical=false,
        p_colorbar1_kwargs(ps)...
    )
    fig
end

end

function paper_figure_setup_reference()
    fig = Figure(;
        size=(double_col_width * 0.35, 0.43 * double_col_width / golden_ratio),
        figure_padding=(2., 2., 2., 6.),
    )

    ax = Axis(fig[1, 1];
        # labels
        xlabel="X label",
        ylabel="Y label",
        xlabelsize=8fontsize_pt,
        ylabelsize=8fontsize_pt,
        xlabelpadding=0.,
        ylabelpadding=0.,
        # ticks style
        xticklabelsize=7fontsize_pt,
        yticklabelsize=7fontsize_pt,
        xticklabelpad=0.,
        yticklabelpad=0.,
        # setup log scale ticks with minors
        xscale=log10,
        xticks=log10ticks(0:3),
        xminorticksvisible=true,
        xminorticks=IntervalsBetween(10),
    )
end
