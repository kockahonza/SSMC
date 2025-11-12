function draw_fr_ext_line!(ax, ls, m, c;
    Ktransform=identity,
    kwargs...
)
    r = 1.0
    line_Ks = map(ls) do l
        if l > 0.5
            4 * l * m * r / c
        else
            m * r / (c * (1 - l))
        end
    end

    lines!(ax, Ktransform.(line_Ks), LeakageScale.ltox.(ls);
        kwargs...
    )

    ax
end
function draw_fr_instab_line!(ax, ls, m, c;
    Ktransform=identity,
    kwargs...
)
    r = 1.0
    line_Ks = map(ls) do l
        if l > 0.5
            m * r / (c * (1 - l))
        else
            missing
        end
    end

    lines!(ax, Ktransform.(line_Ks), LeakageScale.ltox.(ls);
        kwargs...
    )

    ax
end
function draw_fr_lines!(args...; kwargs...)
    draw_fr_ext_line!(args...; kwargs...)
    draw_fr_instab_line!(args...; kwargs...)
end
