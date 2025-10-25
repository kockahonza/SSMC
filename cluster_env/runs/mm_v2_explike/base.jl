using SSMCMain, SSMCMain.ModifiedMiCRM.MinimalModelV2

using Base.Threads
using OhMyThreads

using ProgressMeter
using JLD2

function run_Kl_nospace(;
    logKs=range(-0.5, 3, 100),
    ls=range(0.0, 1.0, 50),
    u0=[1.0, 0.0, 0.0],
    T=1e8,
    tol=100 * eps(),
)
    params = Matrix{Any}(undef, length(logKs), length(ls))
    retcodes = Matrix{ReturnCode.T}(undef, length(logKs), length(ls))
    final_states = Matrix{Vector{Float64}}(undef, length(logKs), length(ls))
    prog = Progress(length(logKs))
    @tasks for i in 1:length(logKs)
        logK = logKs[i]
        for (j, l) in enumerate(ls)
            l = ls[j]
            mmp = MMParams(;
                K=10^logK,
                m=1.0,
                l=l,
                k=0.0,
                c=1.0,
                d=1.0,
            )
            ps = mmp_to_mmicrm(mmp)
            p = make_mmicrm_problem(ps, copy(u0), T)

            s = solve(p, QNDF();
                abstol=tol,
                reltol=tol,
                callback=make_timer_callback(2)
            )

            params[i, j] = ps
            retcodes[i, j] = s.retcode
            final_states[i, j] = s.u[end]
        end
        # @printf "Finished %d out of %d logK runs\n" i length(logKs)
        # flush(stdout)
        next!(prog)
    end
    finish!(prog)

    jldsave("nospace_results.jld2"; logKs, ls, params, retcodes, final_states)

    (; params, retcodes, final_states)
end

function main1()
    logKs = range(-0.5, 3, 100)
    lis = range(0.0, 1.0, 50)
    @time xx = run_explike_Kl_space(logKs, lis, 1000000;
        save_sols=true
    )

    @time jldsave("faf.jld2"; logKs, lis, xx)
end

function main2()
    logKs = range(-0.5, 3, 100)
    ls = range(0.0, 1.0, 50)
    # Spatial setup
    L = 2 # system size in non-dim units
    sN = 10000 # number of spatial points
    epsilon = 1e-5 # initial condition noise amplitude

    dx = L / (sN + 1)
    u0 = clamp.(reduce(hcat, [[1.0, 0.0, 0.0] .+ epsilon .* randn(3) for _ in 1:sN]), 0.0, Inf)

    params = Matrix{Any}(undef, length(logKs), length(ls))
    retcodes = Matrix{ReturnCode.T}(undef, length(logKs), length(ls))
    final_states = Matrix{Matrix{Float64}}(undef, length(logKs), length(ls))
    prog = Progress(length(logKs) * length(ls))

    GC.enable_logging(true)
    @tasks for i in 1:length(logKs)
        logK = logKs[i]
        for (j, l) in enumerate(ls)
            l = ls[j]
            mmp = MMParams(;
                K=10^logK,
                m=1.0,
                l=l,
                k=0.0,
                c=1.0,
                d=1.0,
            )
            sps = SASMMiCRMParams(
                mmp_to_mmicrm(mmp),
                SA[1e-12, 1.0, 1.0],
                make_cartesianspace_smart(1; dx),
                # nthreads()
            )
            sp = make_smmicrm_problem(sps, copy(u0), 1e8)

            tol = 100 * eps()
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

            GC.gc()

            next!(prog)
            flush(stdout)
        end
        # @printf "Finished %d out of %d logK runs\n" i length(logKs)
    end
    finish!(prog)

    jldsave("main2_results.jld2"; logKs, ls, params, retcodes, final_states, L, sN, epsilon)

    (; params, retcodes, final_states)
end

function main4()
    logKs = range(-0.1, 1, 100)
    ls = range(0.0, 1.0, 20)
    # Spatial setup
    L = 2 # system size in non-dim units
    sN = 10000 # number of spatial points
    epsilon = 1e-5 # initial condition noise amplitude

    dx = L / (sN + 1)
    u0 = clamp.(reduce(hcat, [[10.0, 0.0, 0.0] .+ epsilon .* randn(3) for _ in 1:sN]), 0.0, Inf)

    params = Matrix{Any}(undef, length(logKs), length(ls))
    retcodes = Matrix{ReturnCode.T}(undef, length(logKs), length(ls))
    final_states = Matrix{Matrix{Float64}}(undef, length(logKs), length(ls))
    prog = Progress(length(logKs) * length(ls))

    GC.enable_logging(true)
    @tasks for i in 1:length(logKs)
        logK = logKs[i]
        for (j, l) in enumerate(ls)
            l = ls[j]
            mmp = MMParams(;
                K=10^logK,
                m=1.0,
                l=l,
                k=0.0,
                c=1.0,
                d=1.0,
            )
            sps = SASMMiCRMParams(
                mmp_to_mmicrm(mmp),
                SA[1e-12, 1.0, 1.0],
                make_cartesianspace_smart(1; dx),
                # nthreads()
            )
            sp = make_smmicrm_problem(sps, copy(u0), 1e8)

            tol = 1000 * eps()
            s = solve(sp, QNDF();
                dense=false,
                save_everystep=false,
                abstol=tol,
                reltol=tol,
                callback=make_timer_callback(60 * 60)
            )

            params[i, j] = sps
            retcodes[i, j] = s.retcode
            final_states[i, j] = s.u[end]

            GC.gc()

            next!(prog)
            flush(stdout)
        end
        # @printf "Finished %d out of %d logK runs\n" i length(logKs)
    end
    finish!(prog)

    jldsave("main4_results.jld2"; logKs, ls, params, retcodes, final_states, L, sN, epsilon)

    (; params, retcodes, final_states)
end
