function plot_mmicrm_sol(sol; singleax=false, plote=false)
    params = sol.prob.p
    if !isa(params, MMiCRMParams)
        throw(ArgumentError("plot_mmicrm_sol can only plot solutions of MMiCRM problems"))
    end
    Ns, Nr = get_Ns(params)

    fig = Figure()
    if singleax
        strainax = resax = Axis(fig[1, 1])
        if plote
            eax = strainax
        end
    else
        strainax = Axis(fig[1, 1])
        resax = Axis(fig[2, 1])
        if plote
            eax = Axis(fig[3, 1])
        end
    end

    # plot data
    for i in 1:Ns
        lines!(strainax, sol.t, sol[i, :]; label=@sprintf "str %d" i)
    end
    for a in 1:Nr
        lines!(resax, sol.t, sol[Ns+a, :]; label=@sprintf "res %d" a)
    end
    if plote
        lines!(eax, sol.t, calc_E.(sol.u, Ref(params)); label=L"\epsilon")
    end

    if singleax
        axislegend(strainax)
    else
        axislegend(strainax)
        axislegend(resax)
        if plote
            axislegend(eax)
        end
    end
    fig
end
export plot_mmicrm_sol

"""AVOID USING"""
function make_solve_plot_return(args...; kwargs...)
    p = make_mmicrm_smart(args...; kwargs...)
    s = solve(p)
    display(plot_mmicrm_sol(s))
    p, s
end
export make_solve_plot_return
