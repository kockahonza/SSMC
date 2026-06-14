using Makie, ColorSchemes, Colors

function centres_to_edges_closed(xs)
    N = length(xs)
    xs2 = zeros(N+1)
    for i in 1:N-1
        xs2[i+1] = (xs[i] + xs[i+1]) / 2
    end
    xs2[1] = xs[1]
    xs2[end] = xs[end]
    xs2
end

function get_rel_col_20(xx)
    rels = xx ./ sum(xx)

    sp = sortperm(rels)
    
    c = ColorSchemes.tab20[sp[1]]
    s = rels[sp[1]]
    for i in sp[2:end]
        newc = ColorSchemes.tab20[i]

        s += rels[i]
        c = get(cgrad([c, newc]), rels[i] / s)
    end

    c
end
