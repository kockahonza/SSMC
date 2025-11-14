function make_mm_Kl_hm_ax(pos, logKs, ls;
    nyticks=7,
    kwargs...
)
    extra = Dict{Symbol,Any}()
    if !isnothing(ls)
        ext_ls = extrema(ls)
        extra[:yticks] = LeakageScale.exticks(range(1 - ext_ls[2], 1 - ext_ls[1], nyticks))
    end

    ax = Axis(pos;
        ylabel=L"\epsilon",
        xlabel=L"K",
        xscale=log10,
        extra...,
        kwargs...
    )

    if !isnothing(logKs)
        ext = extrema(logKs)
        xlims!(ax, 10^ext[1], 10^ext[2])
    end
    if !isnothing(ls)
        ext = extrema(ls)
        ylims!(ax, LeakageScale.ltox.(ext)...)
    end

    ax
end
make_mm_Kl_hm_ax(pos, ls=nothing; kwargs...) = make_mm_Kl_hm_ax(pos, nothing, ls; kwargs...)

function draw_fr_ext_line!(ax, ls, m, c;
    Ktransform=identity,
    kwargs...
)
    line_Ks = fr_ext_line_K.(ls, m, c)

    lines!(ax, Ktransform.(line_Ks), LeakageScale.ltox.(ls);
        kwargs...
    )

    ax
end
function draw_fr_instab_line!(ax, ls, m, c;
    Ktransform=identity,
    kwargs...
)
    line_Ks = fr_instab_line_K.(ls, m, c)

    lines!(ax, Ktransform.(line_Ks), LeakageScale.ltox.(ls);
        kwargs...
    )

    ax
end
function draw_fr_lines!(args...; kwargs...)
    draw_fr_ext_line!(args...; kwargs...)
    draw_fr_instab_line!(args...; kwargs...)
end
function draw_fr_lines2!(ax, ls, m, c, p, args...; Ktransform=identity, kwargs...)
    ext_line_Ks = fr_ext_line_K.(ls, m, c, args...)
    lines!(ax, Ktransform.(ext_line_Ks), LeakageScale.ltox.(ls);
        kwargs...
    )
    instab_line_Ks = fr_cor1_instab_line_K.(ls, m, c, p, args...)
    lines!(ax, Ktransform.(instab_line_Ks), LeakageScale.ltox.(ls);
        kwargs...
    )
    ax
end
