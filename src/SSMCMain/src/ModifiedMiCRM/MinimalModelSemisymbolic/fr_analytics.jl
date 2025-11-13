function fr_ext_line_K(l, m, c, r=1.0)
    if 0.5 <= l <= 1.0
        4 * l * m * r / c
    elseif 0.0 <= l <= 0.5
        m * r / (c * (1 - l))
    else
        missing
    end
end
function fr_instab_line_K(l, m, c, r=1.0)
    if 0.5 <= l <= 1.0
        m * r / (c * (1 - l))
    else
        missing
    end
end
function fr_cor1_instab_line_K(l, m, c, p, r=1.0)
    # if (1 - 1 / (2 * p)) <= l <= 1.0
    if (p / (1 + p)) <= l <= 1.0
        if p < (1 / (2 * (1 - l)))
            l * m * r / (c * p * (1 - l) * (1 - p * (1 - l)))
        else
            missing
        end
    else
        missing
    end
end

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
