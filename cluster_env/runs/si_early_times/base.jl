using Revise
using SSMCMain, SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.RandomSystems, SSMCMain.ModifiedMiCRM.FFTAnalysis

include("../../../scripts/single_influx.jl")

using Base.Threads, OhMyThreads
using Printf
using ProgressMeter
using JLD2, Geppetto
using Random, Distributions

################################################################################
# Running sims
################################################################################
function do_si_early_time_run(
    out_fname, Klips_to_run, num_runs,
    N, M, DN,
    T, L, sN,
    perturbation_epsilon,
    ;
    save_sols=false,
    save_everystep=false,
    ode_u0=[fill(1.0, N); fill(0.0, M)],
    lsks=10 .^ range(-5, 2, 2000),
    fft_factor=100.,
    pde_solve_maxtime=60 * 60,
    tol=1e-9,
    run_threads=8,
    solver_threads=div(nthreads(), run_threads),
)
    dx = L / sN
    metadata = (;
        Klips_to_run, N, M, DN,
        T, ode_u0, lsks, L, sN, dx,
        perturbation_epsilon,
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
            abstol=tol,
            reltol=tol,
            maxiters=100000,
            # callback=make_timer_callback(10 * 60),
            callback=CallbackSet(make_timer_callback(20 * 60), PositiveDomain(copy(p.u0))),
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
    pde_u0s = Vector{Matrix{Float64}}(undef, total_num_runs)
    pde_retcodes = Vector{ReturnCode.T}(undef, total_num_runs)
    pde_final_states = Vector{Matrix{Float64}}(undef, total_num_runs)
    pde_final_Ts = Vector{Float64}(undef, total_num_runs)
    pde_dom_length = Vector{Float64}(undef, total_num_runs)
    pde_sols = Vector{Any}(undef, total_num_runs)

    println("Starting PDE runs near ODE ss")
    flush(stdout)
    prog1 = Progress(total_num_runs)
    @tasks for ri in 1:total_num_runs
        @set ntasks = run_threads

        ode_ss = ode_final_states[ri]
        u0_ = expand_u0_to_size((sN,), ode_ss)
        u0 = perturb_u0_uniform(N, M, u0_, perturbation_epsilon)

        sps = BSMMiCRMParams(
            params[ri].mmicrm_params,
            params[ri].Ds,
            CartesianSpace{1,Tuple{Periodic}}(SA[dx]),
            solver_threads
        )
        sp = make_smmicrm_problem(sps, u0, T)

        s = solve(sp, QNDF();
            dense=false,
            save_everystep,
            calck=false,
            abstol=tol,
            reltol=tol,
            # callback=make_timer_callback(pde_solve_maxtime),
            callback=CallbackSet(make_fft_callback2(N, sN, dx, fft_factor), make_timer_callback(pde_solve_maxtime), PositiveDomain(copy(sp.u0))),
        )

        pde_u0s[ri] = s.u[1]
        pde_retcodes[ri] = s.retcode
        pde_final_states[ri] = s.u[end]
        pde_final_Ts[ri] = s.t[end]
        pde_dom_length[ri] = get_dominant_lengthscale(get_total_biomass_1d(s.u[end], N), dx)
        pde_sols[ri] = save_sols ? s : nothing

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
        pde_u0s=reshape(pde_u0s, num_runs, num_Klips),
        pde_retcodes=reshape(pde_retcodes, num_runs, num_Klips),
        pde_final_states=reshape(pde_final_states, num_runs, num_Klips),
        pde_final_Ts=reshape(pde_final_Ts, num_runs, num_Klips),
        pde_dom_length=reshape(pde_dom_length, num_runs, num_Klips),
        pde_sols=reshape(pde_sols, num_runs, num_Klips),
    )

    jldsave(out_fname; results...)

    results
end

################################################################################
# Main functions
################################################################################
"""
Taking the Klis from the first single_influx_pdes5_lengths run but also doing l=0.9,
"""
function main1()
    Klips_to_run = [
        (10.782736830354969, 1.0, 1.0),
        (15.176070792347483, 1.0, 1.0),
        (19.59437173444063, 1.0, 1.0),
        (10.782736830354969, 0.9, 1.0),
        (15.176070792347483, 0.9, 1.0),
        (19.59437173444063, 0.9, 1.0),
    ]

    num_runs = 6
    N = 20

    DN = 1e-6

    T = 1e8
    L = 10
    sN = 2500

    perturbation_epsilon = 1e-3

    do_si_early_time_run(
        "data1.jld2", Klips_to_run, num_runs,
        N, N, DN,
        T, L, sN,
        perturbation_epsilon;
        fft_factor=100.,
        run_threads=6,
    )
end

"""
A bigger run, mostly increasing the number of runs at each param value but also
expanding the K range to span the instability region, only doing l=p=1.
"""
function main2()
    Klips_to_run = [(K, 1., 1.) for K in (10 .^ range(0.6, 1.5, 15))]

    num_runs = 30
    N = 20

    DN = 1e-6

    T = 1e8
    L = 10
    sN = 2500

    perturbation_epsilon = 1e-3

    do_si_early_time_run(
        "data2.jld2", Klips_to_run, num_runs,
        N, N, DN,
        T, L, sN,
        perturbation_epsilon;
        fft_factor=100.,
        run_threads=6,
    )
end
"""Same as the above but l=0.99"""
function main3()
    Klips_to_run = [(K, 0.99, 1.) for K in (10 .^ range(0.6, 1.5, 15))]

    num_runs = 30
    N = 20

    DN = 1e-6

    T = 1e8
    L = 10
    sN = 2500

    perturbation_epsilon = 1e-3

    do_si_early_time_run(
        "data2_lowerl.jld2", Klips_to_run, num_runs,
        N, N, DN,
        T, L, sN,
        perturbation_epsilon;
        fft_factor=100.,
        run_threads=6,
    )
end

function main2_smaller()
    Klips_to_run = [(K, 1., 1.) for K in (10 .^ range(0.6, 1.5, 10))]

    num_runs = 18
    N = 20

    DN = 1e-6

    T = 1e8
    L = 10
    sN = 2500

    perturbation_epsilon = 1e-3

    do_si_early_time_run(
        "data2_smaller.jld2", Klips_to_run, num_runs,
        N, N, DN,
        T, L, sN,
        perturbation_epsilon;
        fft_factor=100.,
        run_threads=6,
    )
end

"""
Same as main2 but using the continuous callback and somewhat smaller
"""
function main4()
    Klips_to_run = [(K, 1., 1.) for K in (10 .^ range(0.9, 1.5, 10))]

    num_runs = 30
    N = 20

    DN = 1e-6

    T = 1e8
    L = 10
    sN = 2500

    perturbation_epsilon = 1e-3

    do_si_early_time_run(
        "data4.jld2", Klips_to_run, num_runs,
        N, N, DN,
        T, L, sN,
        perturbation_epsilon;
        fft_factor=100.,
        pde_solve_maxtime=20 * 60,
        run_threads=nthreads(),
        solver_threads=nothing,
    )
end
"""Same as the above but l=0.99"""
function main5()
    Klips_to_run = [(K, 0.99, 1.) for K in (10 .^ range(0.9, 1.5, 10))]

    num_runs = 30
    N = 20

    DN = 1e-6

    T = 1e8
    L = 10
    sN = 2500

    perturbation_epsilon = 1e-3

    do_si_early_time_run(
        "data5.jld2", Klips_to_run, num_runs,
        N, N, DN,
        T, L, sN,
        perturbation_epsilon;
        fft_factor=100.,
        pde_solve_maxtime=20 * 60,
        run_threads=nthreads(),
        solver_threads=nothing,
    )
end

"""
Same as main4 but with a smaller fft_factor
"""
function main6()
    Klips_to_run = [(K, 1., 1.) for K in (10 .^ range(0.9, 1.5, 10))]

    num_runs = 30
    N = 20

    DN = 1e-6

    T = 1e8
    L = 10
    sN = 2500

    perturbation_epsilon = 1e-3

    do_si_early_time_run(
        "data6.jld2", Klips_to_run, num_runs,
        N, N, DN,
        T, L, sN,
        perturbation_epsilon;
        fft_factor=50.,
        pde_solve_maxtime=20 * 60,
        run_threads=nthreads(),
        solver_threads=nothing,
    )
end
