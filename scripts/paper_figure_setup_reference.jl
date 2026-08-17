function fafffffff()
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
