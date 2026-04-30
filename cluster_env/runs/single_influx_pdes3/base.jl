using Revise
using SSMCMain, SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.RandomSystems

include("../../../scripts/single_influx.jl")

using Base.Threads, OhMyThreads
using Printf
using ProgressMeter
using JLD2, Geppetto
using Random, Distributions

################################################################################
# Running PDEs
################################################################################
function run_1d_pde_sim(ps, u0, T, L, sN;
    maxtime=60,
    solver_threads=nothing,
    tol=100000 * eps(),
    solver=QNDF,
)
    dx = L / sN

    sps = BSMMiCRMParams(
        ps.mmicrm_params,
        ps.Ds,
        CartesianSpace{1,Tuple{Periodic}}(SA[dx]),
        solver_threads
    )
    sp = make_smmicrm_problem(sps, u0, T)

    s = solve(sp, solver();
        dense=false,
        save_everystep=false,
        calck=false,
        abstol=tol,
        reltol=tol,
        callback=make_timer_callback(maxtime)
    )

    s
end

################################################################################
# Main functions
################################################################################
"""
Testing
"""
function main1()
    Klis_to_run = [
        (9.319395762340777, 1.0),
        (26.826957952797258, 1.0),
        (77.22449945836259, 1.0),
        (222.29964825261956, 1.0),
        (517.9474679231213, 1.0)
    ]
    N = 20
    M = 20
    DN = 1e-3

    T = 1000000000

    ode_u0 = [fill(1.0, N); fill(0.0, M)]

    lsks = 10 .^ range(-5, 2, 2000)

    L = 15
    sN = 2000

    sp_epsilon = 1e-3

    num_runs = 20

    pde_solve_maxtime = 2 * 60 * 60
    run_threads = 8
    solver_threads = div(nthreads(), run_threads)

    save_filename = "data_test1.jld2"

    metadata = (;
        Klis_to_run, N, M, DN,
        T, ode_u0, lsks, L, sN,
        sp_epsilon,
        num_runs,
        pde_solve_maxtime, run_threads,
    )

    num_Klis = length(Klis_to_run)
    total_num_runs = num_runs * length(Klis_to_run)

    ########################################
    # Generate a bunch of params
    ########################################
    params = Vector{BSMMiCRMParams}(undef, total_num_runs)
    pri = 1
    for (K, li) in Klis_to_run
        for _ in 1:num_run
            params[pri] = get_si_sampler_for_paper(K, li, DN; N, M)
            pri += 1
        end
    end

    ########################################
    # solve ODEs and do linstab
    ########################################
    ode_retcodes = Vector{ReturnCode.T}(undef, total_num_runs)
    ode_final_states = Vector{Vector{Float64}}(undef, total_num_runs)
    ode_final_Ts = Vector{Float64}(undef, total_num_runs)

    linstab_mrls = Vector{Vector{Float64}}(undef, total_num_runs)

    println("Starting ODE runs")
    flush(stdout)
    prog0 = Progress(total_num_runs)
    @tasks for ri in 1:total_num_runs
        # @set ntasks = run_threads
        p = make_mmicrm_problem(params[ri], copy(ode_u0), T)
        s = solve(p)
        ode_retcodes[ri] = s.retcode
        ode_final_states[ri] = s.u[end]
        ode_final_Ts[ri] = s.t[end]

        lsfunc = linstab_make_k_func(p.p, s.u[end];
            returnobj=:maxeval
        )
        linstab_mrls[ri] = real.(lsfunc.(lsks))

        next!(prog0)
        flush(stdout)
    end
    finish!(prog0)
    flush(stdout)
    GC.gc()

    ########################################
    # solve PDEs from perturbed ODE solution
    ########################################
    sp_retcodes = Vector{ReturnCode.T}(undef, total_num_runs)
    sp_final_states = Vector{Matrix{Float64}}(undef, total_num_runs)
    sp_final_Ts = Vector{Float64}(undef, total_num_runs)

    println("Starting PDE runs near ODE ss")
    flush(stdout)
    prog1 = Progress(total_num_runs)
    @tasks for ri in 1:total_num_runs
        @set ntasks = run_threads

        ode_ss = ode_final_states[ri]
        u0_ = expand_u0_to_size((sN,), ode_ss)
        u0 = perturb_u0_uniform(N, M, u0_, sp_epsilon)

        s = run_1d_pde_sim(params[ri], u0, T, L, sN;
            maxtime=pde_solve_maxtime,
            solver_threads,
        )

        sp_retcodes[ri] = s.retcode
        sp_final_states[ri] = s.u[end]
        sp_final_Ts[ri] = s.t[end]

        s = nothing
        GC.gc()

        next!(prog1)
        flush(stdout)
    end
    finish!(prog1)
    flush(stdout)
    GC.gc()

    # reshape results
    results = (;
        metadata,
        params=reshape(params, num_runs, num_Klis),
        ode_retcodes=reshape(ode_retcodes, num_runs, num_Klis),
        ode_final_states=reshape(ode_final_states, num_runs, num_Klis),
        ode_final_Ts=reshape(ode_final_Ts, num_runs, num_Klis),
        linstab_mrls=reshape(linstab_mrls, num_runs, num_Klis),
        sp_retcodes=reshape(sp_retcodes, num_runs, num_Klis),
        sp_final_states=reshape(sp_final_states, num_runs, num_Klis),
        sp_final_Ts=reshape(sp_final_Ts, num_runs, num_Klis),
    )

    jldsave(save_filename; results...)

    results
end
