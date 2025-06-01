function diagram_full(p::AbstractMMiCRMParams)
    Ns, Nr = get_Ns(p)

    g = digraph()

    strains = [(@sprintf "N %d" i) for i in 1:Ns]
    resources = [(@sprintf "R %d" i) for i in 1:Nr]

    for a in 1:Nr
        if p.K[a] != 0.0
            g |> edge("chemostat", resources[a];
                # label=(@sprintf "%.3g" p.c[i, a]),
                weight=string(p.K[a])
            )
        end
    end

    for i in 1:Ns
        for a in 1:Nr
            if p.c[i, a] != 0.0
                g |> edge(resources[a], strains[i];
                    # label=(@sprintf "%.3g" p.c[i, a]),
                    weight=string(p.c[i, a])
                )
                for b in 1:Nr
                    if p.D[i, b, a] != 0.0
                        g |> edge(strains[i], resources[b];
                            # label=(@sprintf "%.3g" p.D[i, a, b]),
                            weight=string(p.D[i, b, a]),
                            style="dashed"
                        )
                    end
                end
            end
        end
    end

    g
end
export diagram_full

function diagram_simple(p::AbstractMMiCRMParams)
    Ns, Nr = get_Ns(p)

    g = digraph()

    strains = [(@sprintf "N %d" i) for i in 1:Ns]
    resources = [(@sprintf "R %d" i) for i in 1:Nr]

    # Add links from outside supply
    for i in 1:Ns
        for a in 1:Nr
            c = p.c[i, a]
            K = p.K[a]
            if (c > 0.0) && (K > 0.0)
                g |> edge("chemostat", strains[i];
                    label=resources[a],
                    # weight=string(c)
                )
            end
        end
    end

    # Add cross feeding links
    for i in 1:Ns
        for j in 1:Ns
            for a in 1:Nr
                # i must produce a
                # and j must eat it (and perhaps optionally, actually grow off of it)
                if p.c[j, a] > 0.0
                    for b in 1:Nr
                        if p.c[i, b] > 0.0 && p.D[i, a, b] > 0.0
                            g |> edge(strains[i], strains[j];
                                label=resources[a],
                            )
                        end
                    end
                end
            end
        end
    end

    g
end
export diagram_simple
