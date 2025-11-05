using SSMCMain, SSMCMain.ModifiedMiCRM.MinimalModelV2

using Base.Threads
using OhMyThreads

using ProgressMeter
using JLD2

function run_Kl_nospace(;
    logKs=range(-0.5, 3, 100),
    ls=range(0.0, 1.0, 50),
    m=1.0,
    c=1.0,
    N0=100.0,
    u0=[N0, 0.0, 0.0],
    T=1e6,
    tol=1000 * eps(),
    maxtime=2,
)
    params = Matrix{Any}(undef, length(logKs), length(ls))
    retcodes = Matrix{ReturnCode.T}(undef, length(logKs), length(ls))
    final_states = Matrix{Vector{Float64}}(undef, length(logKs), length(ls))
    final_Ts = Matrix{Float64}(undef, length(logKs), length(ls))
    prog = Progress(length(logKs))
    @tasks for i in 1:length(logKs)
        logK = logKs[i]
        for (j, l) in enumerate(ls)
            l = ls[j]
            mmp = MMParams(;
                K=10^logK,
                l=l,
                m,
                c,
                k=0.0,
                d=1.0,
            )
            ps = mmp_to_mmicrm(mmp)
            p = make_mmicrm_problem(ps, copy(u0), T)

            s = solve(p, QNDF();
                abstol=tol,
                reltol=tol,
                callback=make_timer_callback(maxtime)
            )

            params[i, j] = ps
            retcodes[i, j] = s.retcode
            final_states[i, j] = s.u[end]
            final_Ts[i, j] = s.t[end]
        end
        next!(prog)
    end
    finish!(prog)

    (; params, retcodes, final_states, final_Ts)
end
function run_Kl_nospace(f::JLD2.JLDFile; kwargs...)
    run_Kl_nospace(;
        logKs=f["logKs"],
        ls=f["ls"],
        m=f["m"],
        c=f["c"],
        N0=f["N0"],
        T=f["T"],
        kwargs...
    )
end

################################################################################
# PDEs version 1 - DN=1e-12, generally low N0
################################################################################
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

"""
Smaller K region and higher initial N
"""
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

################################################################################
# PDEs version 1 - high N0, higher DN
################################################################################
function v2main_highN0_base()
    logKs = range(-0.1, 4, 80)
    ls = LeakageScale.lxrange(0.01, 0.99, 30)

    N0 = 100.0
    DN = 1e-6
    DI = 1.0
    DR = 1.0

    m = 1.0
    c = 1.0

    T = 1e6

    L = 5 # system size in non-dim units
    sN = 5000 # number of spatial points
    epsilon = 1e-5 # initial condition noise amplitude

    ##########

    dx = L / (sN + 1)
    u0 = clamp.(reduce(hcat, [[N0, 0.0, 0.0] .+ epsilon .* randn(3) for _ in 1:sN]), 0.0, Inf)

    params = Matrix{Any}(undef, length(logKs), length(ls))
    retcodes = Matrix{ReturnCode.T}(undef, length(logKs), length(ls))
    final_states = Matrix{Matrix{Float64}}(undef, length(logKs), length(ls))
    final_T = Matrix{Float64}(undef, length(logKs), length(ls))
    prog = Progress(length(logKs) * length(ls))

    @tasks for i in 1:length(logKs)
        logK = logKs[i]
        for (j, l) in enumerate(ls)
            l = ls[j]
            mmp = MMParams(;
                K=10^logK,
                m,
                c,
                l=l,
                k=0.0,
                d=1.0,
            )
            sps = SASMMiCRMParams(
                mmp_to_mmicrm(mmp),
                SA[DN, DI, DR],
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
                callback=make_timer_callback(15 * 60)
            )

            params[i, j] = sps
            retcodes[i, j] = s.retcode
            final_states[i, j] = s.u[end]
            final_T[i, j] = s.t[end]

            next!(prog)
            flush(stdout)
        end
        # @printf "Finished %d out of %d logK runs\n" i length(logKs)
    end
    finish!(prog)

    jldsave("v2main_highN0_base.jld2";
        logKs, ls, N0, DN, DI, DR, m, c, T, L, sN, epsilon,
        params, retcodes, final_states, final_T,
    )


    (; params, retcodes, final_states, final_T)
end

function v2main_highN0_diffs()
    logKs = range(-0.1, 4, 80)
    ls = LeakageScale.lxrange(0.01, 0.99, 30)

    N0 = 100.0
    DN = 1e-6
    DI = 1.0
    DR = 0.1

    m = 10.0
    c = 1.0

    T = 1e6

    L = 5 # system size in non-dim units
    sN = 5000 # number of spatial points
    epsilon = 1e-5 # initial condition noise amplitude

    ##########

    dx = L / (sN + 1)
    u0 = clamp.(reduce(hcat, [[N0, 0.0, 0.0] .+ epsilon .* randn(3) for _ in 1:sN]), 0.0, Inf)

    params = Matrix{Any}(undef, length(logKs), length(ls))
    retcodes = Matrix{ReturnCode.T}(undef, length(logKs), length(ls))
    final_states = Matrix{Matrix{Float64}}(undef, length(logKs), length(ls))
    final_T = Matrix{Float64}(undef, length(logKs), length(ls))
    prog = Progress(length(logKs) * length(ls))

    @tasks for i in 1:length(logKs)
        logK = logKs[i]
        for (j, l) in enumerate(ls)
            l = ls[j]
            mmp = MMParams(;
                K=10^logK,
                m,
                c,
                l=l,
                k=0.0,
                d=1.0,
            )
            sps = SASMMiCRMParams(
                mmp_to_mmicrm(mmp),
                SA[DN, DI, DR],
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
                callback=make_timer_callback(15 * 60)
            )

            params[i, j] = sps
            retcodes[i, j] = s.retcode
            final_states[i, j] = s.u[end]
            final_T[i, j] = s.t[end]

            next!(prog)
            flush(stdout)
        end
        # @printf "Finished %d out of %d logK runs\n" i length(logKs)
    end
    finish!(prog)

    jldsave("v2main_highN0_diffs.jld2";
        logKs, ls, N0, DN, DI, DR, m, c, T, L, sN, epsilon,
        params, retcodes, final_states, final_T,
    )


    (; params, retcodes, final_states, final_T)
end
