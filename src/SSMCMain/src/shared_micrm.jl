################################################################################
# Physicsy bits
################################################################################
"""Calculates the total energy density in the system"""
function calc_E(u,
    p::Union{BasicMiCRM.MiCRMParams{Ns,Nr},ModifiedMiCRM.MMiCRMParams{Ns,Nr}}
) where {Ns,Nr}
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
function param_summary(
    p::Union{BasicMiCRM.MiCRMParams{Ns,Nr},ModifiedMiCRM.MMiCRMParams{Ns,Nr}}
) where {Ns,Nr}
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

################################################################################
# Utility
################################################################################
function plot_linstab_lambdas(ks, lambdas; imthreshold=1e-8)
    fig = Figure()
    ax = Axis(fig[1, 1])
    for li in axes(lambdas, 2)
        lines!(ax, ks, real(lambdas[:, li]);
            color=Cycled(li),
            label=latexstring(@sprintf "\\Re(\\lambda_%d)" li)
        )
        ims = imag(lambdas[:, li])
        mims = maximum(ims)
        if mims > imthreshold
            @info @sprintf "we are getting non-zero imaginary parts, max is %f" mims
            lines!(ax, ks, ims;
                color=Cycled(li),
                linestyle=:dash,
                label=latexstring(@sprintf "\\Im(\\lambda_%d)" li)
            )
        end
    end
    axislegend(ax)
    FigureAxisAnything(fig, ax, lambdas)
end
export plot_linstab_lambdas

