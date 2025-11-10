using SSMCMain, SSMCMain.ModifiedMiCRM.SymCosmo

using Base.Threads
using OhMyThreads

using ProgressMeter
using JLD2

function v1_base()
    logKs = range(-0.1, 4, 80) .+ log10(2.0)
    ls = LeakageScale.lxrange(0.01, 0.99, 30)

    N0 = 100.0
    DN = 1e-6
    DI = 1.0
    DR = 1.0

    m = 1.0
    c = 1.0 / 2.0

    T = 1e6

    L = 5 # system size in non-dim units
    sN = 5000 # number of spatial points
    epsilon = 1e-5 # initial condition noise amplitude

    ##########

    dx = L / (sN + 1)
    u0 = clamp.(reduce(hcat, [[N0, N0, 0.0, 0.0, 0.0] .+ epsilon .* randn(5) for _ in 1:sN]), 0.0, Inf)

    params = Matrix{Any}(undef, length(logKs), length(ls))
    retcodes = Matrix{ReturnCode.T}(undef, length(logKs), length(ls))
    final_states = Matrix{Matrix{Float64}}(undef, length(logKs), length(ls))
    final_Ts = Matrix{Float64}(undef, length(logKs), length(ls))

    prog = Progress(length(logKs) * length(ls))

    @tasks for i in 1:length(logKs)
        logK = logKs[i]
        for (j, l) in enumerate(ls)
            l = ls[j]
            scp = SCParams(;
                K=10^logK,
                m,
                c,
                l=l,
                d=1.0,
            )
            sps = SASMMiCRMParams(
                scp_to_mmicrm(scp),
                SA[DN, DN, DI, DR, DR],
                make_cartesianspace_smart(1; dx),
                # nthreads()
            )
            sp = make_smmicrm_problem(sps, copy(u0), T)

            tol = 10000 * eps()
            s = solve(sp, QNDF();
                dense=false,
                save_everystep=false,
                abstol=tol,
                reltol=tol,
                callback=make_timer_callback(30 * 60)
            )

            params[i, j] = sps
            retcodes[i, j] = s.retcode
            final_states[i, j] = s.u[end]
            final_Ts[i, j] = s.t[end]

            next!(prog)
            flush(stdout)
        end
        # @printf "Finished %d out of %d logK runs\n" i length(logKs)
    end
    finish!(prog)

    jldsave("v1_base1.jld2";
        logKs, ls, N0, DN, DI, DR, m, c, T, L, sN, epsilon,
        params, retcodes, final_states, final_Ts,
    )

    (; params, retcodes, final_states, final_Ts)
end
