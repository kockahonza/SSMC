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
    save_everystep=false,
    kwargs...
)
    dx = L / sN

    sps = BSMMiCRMParams(
        ps.mmicrm_params,
        ps.Ds,
        CartesianSpace{1,Tuple{Periodic}}(SA[dx]),
        solver_threads
    )
    sp = make_smmicrm_problem(sps, u0, T; jac_type=:sparse)

    s = solve(sp, solver();
        dense=false,
        calck=false,
        save_everystep,
        abstol=tol,
        reltol=tol,
        callback=make_timer_callback(maxtime),
        kwargs...
    )

    s
end

################################################################################
# Main function V2. Adding ps!!
################################################################################
function run_v2(
    out_fname, Klips_to_run, num_runs, N, M, DN,
    T, L, sN, sp_epsilon;
    ode_u0=[fill(1.0, N); fill(0.0, M)],
    lsks=10 .^ range(-5, 2, 2000),
    save_pdes_time=false,
    pde_solve_maxtime=2 * 60 * 60,
    run_threads=8,
    solver_threads=div(nthreads(), run_threads),
)
    metadata = (;
        Klips_to_run, N, M, DN,
        T, ode_u0, lsks, L, sN,
        sp_epsilon,
        num_runs,
        pde_solve_maxtime, run_threads,
    )

    num_Klips = length(Klips_to_run)
    total_num_runs = num_runs * length(Klips_to_run)

    ########################################
    # Generate a bunch of params
    ########################################
    params = Vector{BSMMiCRMParams}(undef, total_num_runs)
    pri = 1
    for (K, li, p) in Klips_to_run
        rsg = get_si_sampler_for_paper(K, li, DN; N, M, DR=p)
        for _ in 1:num_runs
            params[pri] = rsg()
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
        p = make_mmicrm_problem(params[ri].mmicrm_params, copy(ode_u0), T)
        s = solve(p, QNDF();
            maxiters=100000,
            callback=make_timer_callback(10 * 60),
        )
        ode_retcodes[ri] = s.retcode
        ode_final_states[ri] = s.u[end]
        ode_final_Ts[ri] = s.t[end]

        lsfunc = linstab_make_k_func(params[ri], s.u[end];
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
    sp_sols = Vector{Any}(undef, total_num_runs)

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
            save_everystep=save_pdes_time,
        )

        sp_retcodes[ri] = s.retcode
        sp_final_states[ri] = s.u[end]
        sp_final_Ts[ri] = s.t[end]
        sp_sols[ri] = s

        next!(prog1)
        flush(stdout)
    end
    finish!(prog1)
    flush(stdout)

    # reshape results
    results = (;
        metadata,
        params=reshape(params, num_runs, num_Klips),
        ode_retcodes=reshape(ode_retcodes, num_runs, num_Klips),
        ode_final_states=reshape(ode_final_states, num_runs, num_Klips),
        ode_final_Ts=reshape(ode_final_Ts, num_runs, num_Klips),
        linstab_mrls=reshape(linstab_mrls, num_runs, num_Klips),
        sp_retcodes=reshape(sp_retcodes, num_runs, num_Klips),
        sp_final_states=reshape(sp_final_states, num_runs, num_Klips),
        sp_final_Ts=reshape(sp_final_Ts, num_runs, num_Klips),
        sp_sols=reshape(sp_sols, num_runs, num_Klips),
    )

    jldsave(out_fname; results...)

    results
end

"""
A small but high quality test run with li=p=1 and aiming for 10, 15 and 20 peaks.
"""
function main1()
    Klips_to_run = [
        (10.782736830354969, 1.0, 1.0),
        (15.176070792347483, 1.0, 1.0),
        (19.59437173444063, 1.0, 1.0),
    ]

    num_runs = 5
    N = 20

    DN = 1e-6

    T = 1000000000
    L = 5
    sN = 5000

    sp_epsilon = 1e-3

    run_v2(
        "data1_li1_p1.jld2",
        Klips_to_run, num_runs, N, N, DN,
        T, L, sN, sp_epsilon;
        save_pdes_time=true,
    )
end
