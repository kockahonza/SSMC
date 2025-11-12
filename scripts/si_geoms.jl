using Revise
using Graphs
using CairoMakie
using GraphMakie
import GraphvizDotLang
using GraphvizDotLang: digraph, subgraph, node, edge, attr, save
using OhMyThreads

function parse_dg6_string(s::String)
    lines = split(s, '\n')

    r1 = r"Graph ([0-9]*), order ([0-9]*)\."
    r2 = r"  ([0-9]*) : ([0-9 ]*);"

    graphs = []
    for i in 1:length(lines)
        l = lines[i]
        intro_lm = match(r1, l)
        if !isnothing(intro_lm)
            n = parse(Int, intro_lm.captures[2])
            g = DiGraph(n)
            for j in 1:n
                bl = lines[i+j]
                bm = match(r2, bl)

                pj = parse(Int, bm.captures[1]) + 1
                @assert j == pj

                if bm.captures[2] != ""
                    for pk in [parse(Int, x) for x in split(bm.captures[2], " ")] .+ 1
                        add_edge!(g, pj, pk)
                    end
                end

            end
            push!(graphs, g)
        end
    end

    graphs
end

function get_bicoloured_graphs(n, m; filter1=true)
    out = Pipe()

    p = run(pipeline(pipeline(`genbg $n $m -q`, `directg -q`, `showg`); stdout=out))
    close(out.in)

    gs = parse_dg6_string(read(out, String))
    if !filter1
        gs
    else
        filter(condition1, gs)
    end
end

condition1(g) =
    all(indegree(g)) do x
        x > 0
    end

function prob(g, n, m)
    sum(2 .* indegree.(Ref(g), 1:n)) + sum(indegree.(Ref(g), n .+ (1:m)))
end

function gv(g, n, m;
    cluster=true
)
    gvg = digraph(; size="10000")

    cn, cr = if cluster
        subgraph(gvg, "cluster n"), subgraph(gvg, "cluster m")
    else
        gvg, gvg
    end

    for i in 1:n
        cn |> node(string(i);
            label="N$i",
        )
    end
    for a in 1:m
        cr |> node(string(a + n);
            label="R$a",
            shape="rect",
        )
    end

    for e in edges(g)
        s = src(e)
        d = dst(e)
        if d <= n
            gvg |> edge(string(s), string(d))
        else
            gvg |> edge(string(s), string(d); style="dashed")
        end
    end

    gvg
end

function plot_graphs(xx, n, m)
    nplots = length(xx)
    fig = Figure(;
        size=(400, 200 * nplots)
    )
    for i in 1:nplots
        ax = Axis(fig[i, 1])
        graphplot!(ax, xx[i];
            node_color=[fill(:red, n); fill(:blue, m)]
        )
    end

    fig
end

function many_gvs(fname, gs, n, m)
    td = mktempdir()
    fns = []
    for i in 1:length(gs)
        gvg = gv(gs[i], n, m)
        fn = joinpath(td, "graph_$i.pdf")
        GraphvizDotLang.save(gvg, fn; format="pdf")
        push!(fns, fn)
    end

    cmd = `pdfunite $fns $fname`
    run(cmd)
    rm(td; recursive=true)

    fname
end
