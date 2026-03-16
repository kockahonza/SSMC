module GraphAnalysis

using Reexport
@reexport using ..ModifiedMiCRM

using EnumX
using Graphs
using MetaGraphsNext

################################################################################
# Making energy flow graphs, in these every node represents a form of energy (
# mainly strains and resources) and every edge is an energy rate between those
################################################################################
abstract type FGNodeType end
abstract type GNodeType <: FGNodeType end
struct Source <: GNodeType end
struct Sink <: GNodeType end
struct R <: GNodeType
    i::Int
end
struct S <: GNodeType
    i::Int
end
struct M <: FGNodeType
    i::Int
end
export FGNodeType, GNodeType, Source, Sink, R, S, M

function make_eflowgraph_simple(ps, ss;
    clamp_zero=false,
    default_weight=0
)
    Ns, Nr = get_Ns(ps)

    if clamp_zero
        ss = clamp_ss(ss)
    end

    mg = MetaGraph(
        DiGraph();
        label_type=FGNodeType,
        vertex_data_type=@NamedTuple{name::String, val::Float64, desc::String},
        edge_data_type=Float64,
        graph_data=(;
            Ns, Nr, ps, ss
        ),
        weight_function=identity,
        default_weight,
    )

    # first add
    mg[Source()] = (; name="Source", val=1.0, desc="source")
    mg[Sink()] = (; name="Sink", val=-Inf, desc="sink")
    for i in 1:Ns
        mg[S(i)] = (
            name="N_$i",
            val=ss[i] / ps.g[i],
            desc="strain $i"
        )
    end
    for a in 1:Nr
        mg[R(a)] = (
            name="R_$a",
            val=ss[Ns+a] * ps.w[a],
            desc="resource $a"
        )
    end

    # add edges to/from the Sink/Source
    for a in 1:Nr
        if !iszero(ps.K[a])
            # Note that the source value is set to 1
            mg[Source(), R(a)] = ps.K[a] * ps.w[a]
        end
        mg[R(a), Sink()] = ps.r[a] * ss[Ns+a] * ps.w[a]
    end
    for i in 1:Ns
        mg[S(i), Sink()] = ps.m[i] * ss[i]
    end

    # add the metabolic edges, the complex bit
    for i in 1:Ns
        for a in 1:Nr
            c = ps.c[i, a]
            if !iszero(c)
                C = ss[i] * ps.c[i, a] * ss[Ns+a] * ps.w[a]
                mg[R(a), S(i)] = C

                l = ps.l[i, a]
                if !iszero(l)
                    sum_of_Ds = 0.0
                    for b in 1:Nr
                        D = ps.D[i, b, a]
                        if !iszero(D)
                            sum_of_Ds += D

                            kk = l * D * C
                            s = S(i)
                            d = R(b)
                            if haskey(mg, s, d)
                                mg[s, d] = kk + mg[s, d]
                            else
                                mg[s, d] = kk
                            end
                        end
                    end
                    if sum_of_Ds < 1.0
                        mg[S(i), Sink()] += l * (1 - sum_of_Ds) * C
                    end
                end
            end
        end
    end

    mg
end
export make_eflowgraph_simple

function remove_env(mg)
    nmg = copy(mg)
    rem_vertex!(nmg, code_for(nmg, Source()))
    rem_vertex!(nmg, code_for(nmg, Sink()))
    nmg
end
export remove_env

################################################################################
# Making a simplified version which only includes the strains
################################################################################
"""
Takes the result of make_eflowgraph_simple
"""
function make_strain_ecouple_graph(mg::MetaGraph;
    default_weight=mg.default_weight,
    skip_self=false,
    threshold=-Inf,
)
    Ns = mg[].Ns
    Nr = mg[].Nr

    sg = MetaGraph(
        DiGraph();
        label_type=GNodeType,
        vertex_data_type=Float64,
        edge_data_type=Float64,
        graph_data=(;
            Ns, Nr, mg
        ),
        weight_function=identity,
        default_weight
    )

    # add the strains
    for i in 1:Ns
        sg[S(i)] = mg[S(i)].val
    end

    # add the energy couplings
    for i in 1:Ns
        for r in outneighbor_labels(mg, S(i))
            if !(r isa R)
                continue
            end
            r_neighbors = outneighbor_labels(mg, r)
            r_total_outflux = sum(n -> mg[r, n], r_neighbors)
            for s in r_neighbors
                if !(s isa S)
                    continue
                end
                if skip_self && (s == S(i))
                    continue
                end
                connection_val = (mg[r, s] / r_total_outflux) * mg[S(i), r]
                if connection_val > threshold
                    if haskey(sg, S(i), s)
                        sg[S(i), s] += connection_val
                    else
                        sg[S(i), s] = connection_val
                    end
                end
            end
        end
    end

    sg
end
export make_strain_ecouple_graph

function make_strain_ecouple_matrix(ps, ss;
    threshold=-Inf,
    clamp_zero=false,
)
    Ns, Nr = get_Ns(ps)

    if clamp_zero
        ss = clamp_ss(ss)
    end

    strains_produce_mat = zeros(Ns, Nr)
    resources_go_to_props_mat = zeros(Nr, Ns)

    for i in 1:Ns
        for a in 1:Nr
            if !iszero(ps.c[i, a])
                C = ss[i] * ps.c[i, a] * ss[Ns+a] * ps.w[a]

                resources_go_to_props_mat[a, i] = C

                l = ps.l[i, a]
                if !iszero(l)
                    for b in 1:Nr
                        D = ps.D[i, b, a]
                        if !iszero(D)
                            kk = l * D * C
                            strains_produce_mat[i, a] += kk
                        end
                    end
                end
            end
        end
    end

    for a in 1:Nr
        xx = @view resources_go_to_props_mat[a, :]
        ss = sum(xx)
        xx ./= ss
    end

    ecouple_mat = zeros(Ns, Ns)
    for i in 1:Ns
        for a in 1:Nr
            if !iszero(strains_produce_mat[i, a])
                for j in 1:Ns
                    if !iszero(resources_go_to_props_mat[a, j])
                        ecouple_mat[i, j] += strains_produce_mat[i, a] * resources_go_to_props_mat[a, j]
                    end
                end
            end
        end
    end

    # strains_produce_mat, resources_go_to_props_mat, ecouple_mat
    ecouple_mat
end
export make_strain_ecouple_matrix

end
