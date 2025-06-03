"""
Use this one! sfss means semifull steady state - shows both strains and
resources and needs a steady state. This will show all strains which hold
at least strain_threshold worth of energy and all the resources that at least
one strain uses at a rate of at least edge_threshold. Everything shown is in
energies and their rates.
"""
function diagram_sfss_v3(p::AbstractMMiCRMParams, ss=nothing;
    strain_threshold=-Inf,
    colorscale=log10,
    node_colormap=colorschemes[:viridis],
    node_colorrange=nothing,
    node_colorscale=colorscale,
    edge_threshold=0.0,
    edge_colormap=colorschemes[:viridis],
    edge_colorrange=nothing,
    edge_colorscale=colorscale,
)
    Ns, Nr = get_Ns(p)
    if isnothing(ss)
        ss = fill(1.0, Ns + Nr)
    end

    # prep graph
    g = digraph()
    node_names = vcat(["N" * string(i) for i in 1:Ns], ["R" * string(a) for a in 1:Nr])

    # make the C matrix first to decide on which resources to include
    C = Matrix{Float64}(undef, Nr, Ns)
    for i in 1:Ns
        for a in 1:Nr
            C[a, i] = ss[i] * p.c[i, a] * ss[Ns+a] * p.w[a]
        end
    end

    # find the nodes we care about
    strains_to_include = Int[]
    strain_E_vals = Float64[]
    resources_to_include = Int[]
    for i in 1:Ns
        val = ss[i] / p.g[i]
        if val > strain_threshold
            push!(strains_to_include, i)
            push!(strain_E_vals, val)
            # include resource that this strain uses
            for a in 1:Nr
                if C[a, i] > edge_threshold
                    if !(a in resources_to_include)
                        push!(resources_to_include, a)
                    end
                end
            end
        end
    end
    resource_E_vals = [ss[Ns+a] * p.w[a] for a in resources_to_include]

    strain_cvals = map(node_colorscale, strain_E_vals)
    resource_cvals = map(node_colorscale, resource_E_vals)
    if isnothing(node_colorrange)
        node_E_val_min, node_E_val_max = extrema(vcat(strain_cvals, resource_cvals))
        node_E_val_delta = node_E_val_max - node_E_val_min
    else
        node_E_val_min, node_E_val_max = node_colorrange
        node_E_val_delta = node_E_val_max - node_E_val_min
    end
    node_cfunc = if !iszero(node_E_val_delta)
        x -> (x - node_E_val_min) / node_E_val_delta
    else
        _ -> 1.0
    end

    # calculate edges/arrows we care about
    edges = []
    for i in strains_to_include
        for a in resources_to_include
            res_to_strain = C[a, i]
            if res_to_strain > edge_threshold
                push!(edges, (
                    node_names[Ns+a], node_names[i], res_to_strain
                ))
            end
            strain_to_res = 0.0
            for b in 1:Nr
                strain_to_res += p.D[i, a, b] * p.l[i, b] * ss[i] * p.c[i, b] * ss[Ns+b] * p.w[b]
            end
            if strain_to_res > edge_threshold
                push!(edges, (
                    node_names[i], node_names[Ns+a], strain_to_res
                ))
            end
        end
    end
    # add chemostat edges
    for a in resources_to_include
        val = p.K[a] * p.w[a]
        if val > edge_threshold
            push!(edges, (
                "chemostat", node_names[Ns+a], val
            ))
        end
    end

    edge_cvals = [edge_colorscale(v) for (_, _, v) in edges]
    if isnothing(edge_colorrange)
        edge_E_val_min, edge_E_val_max = extrema(edge_cvals)
        edge_E_val_delta = edge_E_val_max - edge_E_val_min
    else
        edge_E_val_min, edge_E_val_max = edge_colorrange
        edge_E_val_delta = edge_E_val_max - edge_E_val_min
    end
    edge_cfunc = if !iszero(edge_E_val_delta)
        x -> (x - edge_E_val_min) / edge_E_val_delta
    else
        _ -> 1.0
    end

    # draw the nodes
    g |> node("chemostat";
        label="chemostat",
        shape="rect"
        # color="#" * hex(get(node_colormap, 1.0))
    )
    for (i, val, cval) in zip(strains_to_include, strain_E_vals, strain_cvals)
        label = @sprintf "N_%d\n%.3g" i val
        g |> node(node_names[i];
            label,
            shape="ellipse",
            color="#" * hex(get(node_colormap, node_cfunc(cval)))
        )
    end
    for (a, val, cval) in zip(resources_to_include, resource_E_vals, resource_cvals)
        label = @sprintf "R_%d\n%.3g" a val
        g |> node(node_names[Ns+a];
            label,
            shape="rect",
            color="#" * hex(get(node_colormap, node_cfunc(cval)))
        )
    end

    # draw the edges
    for ((out, in, val), cval) in zip(edges, edge_cvals)
        g |> edge(out, in;
            label=(@sprintf "%.3g" val),
            color="#" * hex(get(edge_colormap, edge_cfunc(cval)))
        )
    end


    g
end
export diagram_sfss_v3

################################################################################
# Work in progress
################################################################################
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

function diagram_sfss_v1(p::AbstractMMiCRMParams, ss;
    node_threshold=0.0,
    node_colormap=colorschemes[:viridis],
    node_colorscale=identity,
    edge_threshold=0.0,
    edge_colormap=colorschemes[:viridis],
)
    Ns, Nr = get_Ns(p)

    g = digraph()

    # prep info about which strains, resources and arrows to draw
    strains = []
    for i in 1:Ns
        strain_E_val = ss[i] / p.g[i]
        if abs(strain_E_val) > node_threshold
            push!(strains, (i, strain_E_val))
        end
    end

    resource_present = falses(Nr)
    edges = []
    for (i, _) in strains
        for a in 1:Nr
            # this will be in energy (density) per time
            res_usage_E_val = ss[i] * p.c[i, a] * ss[Ns+a] * p.w[a]
            if abs(res_usage_E_val) > edge_threshold
                push!(edges,
                    ("R" * string(a), "N" * string(i), res_usage_E_val)
                )
                resource_present[a] = true

                # and add any byproducts
                l = p.l[i, a]
                if l > 0.0
                    for b in 1:Nr
                        res_byprod_E_val = l * res_usage_E_val * p.D[i, b, a]
                        if abs(res_byprod_E_val) > edge_threshold
                            push!(edges,
                                ("N" * string(i), "R" * string(b), res_byprod_E_val)
                            )
                            resource_present[b] = true
                        end
                    end
                end
            end
        end
    end
    resources = []
    for a in 1:Nr
        if resource_present[a]
            push!(resources, (a, ss[Ns+a] * p.w[a]))
        end
    end

    # add chemostat arrows
    for (a, _) in resources
        if p.K[a] > 0.0
            edge_E_cur_val = p.K[a] * p.w[a]
            if abs(edge_E_cur_val) > edge_threshold
                push!(edges, ("chemostat", "R" * string(a), edge_E_cur_val))
            end
        end
    end

    # draw nodes
    strain_E_val_min, strain_E_val_max = extrema(x -> x[2], strains)
    resouce_E_val_min, resouce_E_val_max = extrema(x -> x[2], resources)
    node_E_val_min = min(strain_E_val_min, resouce_E_val_min)
    node_E_val_max = max(strain_E_val_max, resouce_E_val_max)
    node_cfunc(x) = node_colorscale((x - node_E_val_min) / (node_E_val_max - node_E_val_min))

    for (i, strain_E_val) in strains
        g |> node("N" * string(i);
            label=(@sprintf "N_%d\n%.3g" i strain_E_val),
            color="#" * hex(get(node_colormap, node_cfunc(strain_E_val)))
        )
    end
    for (a, resource_E_val) in resources
        g |> node("R" * string(a);
            label=(@sprintf "R_%d\n%.3g" a resource_E_val),
            color="#" * hex(get(node_colormap, node_cfunc(resource_E_val)))
        )
    end

    # draw edges
    edges_min, edges_max = extrema(x -> x[3], edges)
    edges_cfunc(x) = (x - edges_min) / (edges_max - edges_min)

    for (outlabel, inlabel, val) in edges
        g |> edge(outlabel, inlabel;
            label=(@sprintf "%.3g" val),
            weight=string(val),
            color="#" * hex(get(edge_colormap, edges_cfunc(val)))
        )
    end

    g
end
export diagram_sfss_v1

function diagram_sfss_v2(p::AbstractMMiCRMParams, ss;
    node_threshold=-Inf,
    node_colormap=colorschemes[:viridis],
    edge_threshold=-Inf,
    edge_colormap=colorschemes[:viridis],
)
    Ns, Nr = get_Ns(p)

    # prep graph
    g = digraph()
    node_names = vcat(["N" * string(i) for i in 1:Ns], ["R" * string(a) for a in 1:Nr])

    # find the nodes we care about
    strains_to_include = Int[]
    strain_E_vals = Float64[]
    for i in 1:Ns
        val = ss[i] / p.g[i]
        if val > node_threshold
            push!(strains_to_include, i)
            push!(strain_E_vals, val)
        end
    end
    resources_to_include = Int[]
    resource_E_vals = Float64[]
    for a in 1:Nr
        val = ss[Ns+a] * p.w[a]
        offset = val - (p.K[a] / p.r[a]) * p.w[a]
        if abs(offset) > node_threshold
            push!(resources_to_include, a)
            push!(resource_E_vals, val)
        end
    end
    node_E_val_min, node_E_val_max = extrema(vcat(strain_E_vals, resource_E_vals))
    node_cfunc(x) = (x - node_E_val_min) / (node_E_val_max - node_E_val_min)

    # calculate edges/arrows we care about
    edges = []
    for i in strains_to_include
        for a in resources_to_include
            res_to_strain = ss[i] * p.c[i, a] * ss[Ns+a] * p.w[a]
            if res_to_strain > edge_threshold
                push!(edges, (
                    node_names[Ns+a], node_names[i], res_to_strain
                ))
            end
            strain_to_res = 0.0
            for b in 1:Nr
                strain_to_res += p.D[i, a, b] * p.l[i, b] * ss[i] * p.c[i, b] * ss[Ns+b] * p.w[b]
            end
            if strain_to_res > edge_threshold
                push!(edges, (
                    node_names[i], node_names[Ns+a], strain_to_res
                ))
            end
        end
    end
    # add chemostat edges
    for a in resources_to_include
        val = p.K[a] * p.w[a]
        if val > edge_threshold
            push!(edges, (
                "chemostat", node_names[Ns+a], val
            ))
        end
    end
    edge_E_val_min, edge_E_val_max = extrema(x -> x[3], edges)
    edge_cfunc(x) = (x - edge_E_val_min) / (edge_E_val_max - edge_E_val_min)

    # draw the nodes
    for (i, val) in zip(strains_to_include, strain_E_vals)
        label = @sprintf "N_%d\n%.3g" i val
        g |> node(node_names[i];
            label,
            color="#" * hex(get(node_colormap, node_cfunc(val)))
        )
    end
    for (a, val) in zip(resources_to_include, resource_E_vals)
        label = @sprintf "R_%d\n%.3g" a val
        g |> node(node_names[Ns+a];
            label,
            color="#" * hex(get(node_colormap, node_cfunc(val)))
        )
    end

    # draw the edges
    for (out, in, val) in edges
        g |> edge(out, in;
            label=(@sprintf "%.3g" val),
            color="#" * hex(get(edge_colormap, edge_cfunc(val)))
        )
    end


    g
end
export diagram_sfss_v2
