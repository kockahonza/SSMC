# figure sizes for publication
single_col_width = 324 * 1.051437737 # corresponds to 511 pdf pt which is ~9cm
double_col_width = single_col_width * 2
golden_ratio = 1.618
pt_per_mm_ratio = 3.779527559

fig_fontsize = 8

default_fig_kwargs = (;
    fontsize=8,
    markersize=6,
    figure_padding=(2, 2, 2, 2)
)

# colors to use everywhere
module PaperColors
using ColorSchemes
using Makie

extinct1() = ColorSchemes.Blues[5]
extinct2() = ColorSchemes.Blues[7]
extinct3() = ColorSchemes.Blues[9]
stable1() = ColorSchemes.Oranges[4]
stable2() = ColorSchemes.Oranges[5]
unstable1() = ColorSchemes.Greens[4]
unstable2() = ColorSchemes.Greens[5]
other() = ColorSchemes.Dark2_4[end]

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

end
