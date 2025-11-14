module SingleInflux

using Reexport
@reexport using ..ModifiedMiCRM

using Graphs

function make_si_li1_graph(ps::AbstractMMiCRMParams)
    Ns, Nr = get_Ns(ps)

    if count(!iszero, ps.K) != 1
        throw(ArgumentError("This is not a single influx system"))
    end
    a_si = findfirst(!iszero, ps.K)

    g = DiGraph(Ns + Nr)

    for i in 1:Ns
        if !iszero(ps.c[i, a_si]) && !iszero(ps.l[i, a_si])
            for b in 1:Nr
                if !iszero(ps.D[i, b, a_si])
                    add_edge!(g, i, Ns + b)
                end
            end
        end
    end

end

end
