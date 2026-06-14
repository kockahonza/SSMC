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
    abstol=tol,
    reltol=tol,
    pde_solver=TRBDF2,
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

    s = solve(sp, pde_solver();
        dense=false,
        save_everystep=false,
        calck=false,
        abstol,
        reltol,
        callback=CallbackSet(make_timer_callback(maxtime), PositiveDomain(copy(u0); save=false)),
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
    pde_solve_maxtime=2 * 60 * 60,
    run_threads=8,
    solver_threads=div(nthreads(), run_threads),
    kwargs...
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
        s = solve(p, TRBDF2();
            maxiters=100000,
            callback=CallbackSet(make_timer_callback(10 * 60), PositiveDomain(copy(ode_u0); save=false)),
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
            kwargs...
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
        params=reshape(params, num_runs, num_Klips),
        ode_retcodes=reshape(ode_retcodes, num_runs, num_Klips),
        ode_final_states=reshape(ode_final_states, num_runs, num_Klips),
        ode_final_Ts=reshape(ode_final_Ts, num_runs, num_Klips),
        linstab_mrls=reshape(linstab_mrls, num_runs, num_Klips),
        sp_retcodes=reshape(sp_retcodes, num_runs, num_Klips),
        sp_final_states=reshape(sp_final_states, num_runs, num_Klips),
        sp_final_Ts=reshape(sp_final_Ts, num_runs, num_Klips),
    )

    jldsave(out_fname; results...)

    results
end

"""
First small test to see both correctness and viability for computational power
"""
function main1()
    Klips_to_run = [
        (6.0, 0.999, 1.),
        (10.0, 0.999, 1.),
        (17.0, 0.999, 1.),
    ]

    num_runs = 16
    N = 20

    DN = 1e-6

    T = 1000000000
    L = 10
    sN = 2500

    sp_epsilon = 1e-3

    run_v2(
        "data1.jld2",
        Klips_to_run, num_runs, N, N, DN,
        T, L, sN, sp_epsilon;
        run_threads=16,
        pde_solve_maxtime=10 * 60 * 60,
    )
end

"""
Changing p right away, also further increasing run times to reach ss
"""
function main2()
    Klips_to_run = [
        (10.0, 0.999, 1.),
        (10.0, 0.999, 0.1),
        (10.0, 0.999, 0.01),
    ]

    num_runs = 8
    N = 20

    DN = 1e-6

    T = 1000000000
    L = 10
    sN = 2000

    sp_epsilon = 1e-3

    run_v2(
        "data2.jld2",
        Klips_to_run, num_runs, N, N, DN,
        T, L, sN, sp_epsilon;
        run_threads=16,
        pde_solve_maxtime=20 * 60 * 60,
    )
end

"""
Changing p right away, using larger abstol and QNDF!!
"""
function main3()
    Klips_to_run = [
        (10.0, 0.999, 1.),
        (10.0, 0.999, 0.31622776601683794),
        (10.0, 0.999, 0.1),
        (10.0, 0.999, 0.03162277660168379),
        (10.0, 0.999, 0.01),
    ]

    num_runs = 20
    N = 20

    DN = 1e-6

    T = 1e8
    L = 10
    sN = 2000

    sp_epsilon = 1e-3

    run_v2(
        "data3.jld2",
        Klips_to_run, num_runs, N, N, DN,
        T, L, sN, sp_epsilon;
        #
        run_threads=128,
        solver_threads=1,
        #
        pde_solver=QNDF,
        abstol=1e-7,
        reltol=1e-9,
        pde_solve_maxtime=10 * 60 * 60,
    )
end
