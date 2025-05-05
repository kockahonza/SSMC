"""Calculates the total energy density in the system"""
function calc_E(u, p::AbstractMMiCRMParams)
    Ns, Nr = get_Ns(p)
    N = @view u[1:Ns]
    R = @view u[Ns+1:Ns+Nr]

    E = 0.0
    for i in 1:Ns
        E += N[i] / p.g[i]
    end
    for a in 1:Nr
        E += p.w[a] * R[a]
    end
    E
end
export calc_E

"""Print physicsy info about a system"""
function param_summary(p::AbstractMMiCRMParams)
    Ns, Nr = get_Ns(p)
    for a in 1:Nr
        println(@sprintf "Resource %d iseq is %f" a p.K[a] / p.r[a])
    end

    tab = Matrix{String}(undef, Ns, Nr + 2)
    for i in 1:Ns
        tab[i, 1] = @sprintf "%d" i
        for a in 1:Nr
            tab[i, 1+a] = @sprintf "%f" ((p.K[a] / p.r[a]) * p.w[a] * (1 - p.l[a]) * p.c[i, a])
        end
        tab[i, end] = @sprintf "%f" p.m[i]
    end
    pretty_table(tab; header=vcat("Strain i", [@sprintf "ss prod by %d" a for a in 1:Nr], "Upkeep"))
end
param_summary(p::ODEProblem) = param_summary(p.p)
export param_summary
