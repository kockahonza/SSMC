using Revise
using SSMCMain, SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.RandomSystems

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
# Generating random systems
################################################################################
function gen_many_mms1(N;
    K=1.0,
    r=1.0,
    # internal params for every MM replicate
    l=Dirac(1.0),        # (1-l) of I will go to MM, l * (1 - k) will go towards this MMs R, l * k will go to other communities
    m=Dirac(1.0),
    c=Dirac(1.0),
    differentd=nothing,  # will use exactly the same value as for c if nothing
    # weak links
    k=Dirac(0.0),        # proportional of leakage that goes to other communities' internal resources
    B=Dirac(1),          # number of "outgoing links" for every MM replicate
    even_links=true,
    # diffusions
    DI=1.0,
    DR=Dirac(1.0),
    DN=Dirac(1e-6),
)

    S = N
    M = N + 1

    # simple strain params
    pm = Vector{Float64}(undef, S)
    for i in 1:S
        pm[i] = rand(m)
    end

    # simple resource params
    pr = fill(r, M)
    pK = zeros(M)
    pK[1] = K

    # metabolic params
    pc = zeros(S, M)
    pl = zeros(S, M)
    pD = zeros(S, M, M)
    for i in 1:S
        iR = 1 + i
        pc[i, 1] = rand(c)
        pl[i, 1] = rand(l)

        if isnothing(differentd)
            pc[i, iR] = pc[i, 1]
        else
            pc[i, iR] = rand(differentd)
        end

        lk = rand(k)
        lB = min(rand(B), M - 2)

        lD = zeros(M)
        lD[iR] = iszero(lB) ? 1.0 : 1.0 - lk

        possible_byprod_resources = filter(!=(iR), 2:M)
        byprod_resources = sample(possible_byprod_resources, lB; replace=false)
        byprod_split = if even_links
            fill(lk / lB, lB)
        else
            xaxa = rand(lB)
            lk .* xaxa ./ sum(xaxa)
        end
        for xxi in 1:lB
            lD[byprod_resources[xxi]] = byprod_split[xxi]
        end

        (@view pD[i, :, 1]) .= lD

    end

    # diffusions
    pDs = Vector{Float64}(undef, S + M)
    for i in 1:S
        pDs[i] = rand(DN)
    end
    pDs[S+1] = DI
    for a in 1:N
        pDs[S+1+a] = rand(DR)
    end

    ps = BMMiCRMParams(fill(1.0, S), fill(1.0, M), pm, pK, pr, pl, pc, pD)
    BSMMiCRMParams(ps, pDs)
end

################################################################################
# Main functions
################################################################################
"""
Seem to get occasional multistability, but no instabilities at all - dispersion relations look about right but don't quite cross 0.
"""
function main1()
    N = 50
    S = N
    M = N + 1
    genparams = (;
        K=50.0,
        #
        l=Dirac(1.0),
        m=base10_lognormal(0.0, 0.1),
        c=base10_lognormal(0.0, 0.1),
        #
        k=Dirac(0.3),
        B=Binomial(N, 5 / N),
        #
        DR=Dirac(0.1)
    )

    T = 1000000000

    ode_u0 = [fill(1.0, S); fill(0.0, M)]

    lsks = 10 .^ range(-5, 2, 2000)

    L = 5
    sN = 5000

    sp_epsilon = 1e-3

    nh_baseline = 100.0
    nh_numwaves = 50
    nh_maxamp = 100.0

    num_runs = 50

    pde_solve_maxtime = 10 * 60 * 60
    run_threads = 8
    solver_threads = div(nthreads(), run_threads)

    save_filename = "data1.jld2"

    metadata = (;
        N, S, M, genparams, T, ode_u0, lsks, L, sN,
        sp_epsilon,
        nh_numwaves, nh_baseline, nh_maxamp,
        num_runs,
        pde_solve_maxtime, run_threads,
    )

    ########################################
    # Generate a bunch of params
    ########################################
    params = Vector{BSMMiCRMParams}(undef, num_runs)
    @tasks for ri in 1:num_runs
        params[ri] = gen_many_mms1(N; genparams...)
    end

    ########################################
    # solve ODEs and do linstab
    ########################################
    ode_retcodes = Vector{ReturnCode.T}(undef, num_runs)
    ode_final_states = Vector{Vector{Float64}}(undef, num_runs)
    ode_final_Ts = Vector{Float64}(undef, num_runs)

    linstab_mrls = Vector{Vector{Float64}}(undef, num_runs)

    println("Starting ODE runs")
    flush(stdout)
    @tasks for ri in 1:num_runs
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
    end
    GC.gc()

    ########################################
    # solve PDEs from perturbed ODE solution
    ########################################
    sp_retcodes = Vector{ReturnCode.T}(undef, num_runs)
    sp_final_states = Vector{Matrix{Float64}}(undef, num_runs)
    sp_final_Ts = Vector{Float64}(undef, num_runs)

    println("Starting PDE runs near ODE ss")
    flush(stdout)
    prog1 = Progress(num_runs)
    @tasks for ri in 1:num_runs
        @set ntasks = run_threads

        ode_ss = ode_final_states[ri]
        u0_ = expand_u0_to_size((sN,), ode_ss)
        u0 = perturb_u0_uniform(S, M, u0_, sp_epsilon)

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
    GC.gc()

    ########################################
    # solve PDEs from perturbed ODE solution
    ########################################
    nh_u0s = Vector{Matrix{Float64}}(undef, num_runs)
    nh_retcodes = Vector{ReturnCode.T}(undef, num_runs)
    nh_final_states = Vector{Matrix{Float64}}(undef, num_runs)
    nh_final_Ts = Vector{Float64}(undef, num_runs)

    println("Starting PDE runs from non-homogeneous initial condition")
    flush(stdout)
    prog2 = Progress(num_runs)
    @tasks for ri in 1:num_runs
        @set ntasks = run_threads

        u0 = fill(0.0, S + M, sN)
        for i in 1:S
            xxaa = @view u0[i, :]
            xxaa .= nh_baseline
            add_1d_many_sines2!(xxaa, nh_numwaves, nh_maxamp / nh_maxamp; nmax=10)
        end
        clamp!(u0, 0.0, Inf)
        nh_u0s[ri] = u0

        s = run_1d_pde_sim(params[ri], u0, T, L, sN;
            maxtime=pde_solve_maxtime,
            solver_threads,
        )

        nh_retcodes[ri] = s.retcode
        nh_final_states[ri] = s.u[end]
        nh_final_Ts[ri] = s.t[end]

        s = nothing
        GC.gc()

        next!(prog2)
        flush(stdout)
    end
    GC.gc()

    results = (;
        metadata,
        params,
        ode_retcodes,
        ode_final_states,
        ode_final_Ts,
        linstab_mrls,
        sp_retcodes,
        sp_final_states,
        sp_final_Ts,
        nh_u0s,
        nh_retcodes,
        nh_final_states,
        nh_final_Ts,
    )

    jldsave(save_filename; results...)

    results
end

"""
Just as main1 but at a lower K in hopes of being closer to extinction and thus having a higher chance of spatial instability
"""
function main2()
    N = 50
    S = N
    M = N + 1
    genparams = (;
        K=10.0,
        #
        l=Dirac(1.0),
        m=base10_lognormal(0.0, 0.1),
        c=base10_lognormal(0.0, 0.1),
        #
        k=Dirac(0.3),
        B=Binomial(N, 5 / N),
        #
        DR=Dirac(0.1)
    )

    T = 1000000000

    ode_u0 = [fill(1.0, S); fill(0.0, M)]

    lsks = 10 .^ range(-5, 2, 2000)

    L = 5
    sN = 5000

    sp_epsilon = 1e-3

    nh_baseline = 100.0
    nh_numwaves = 50
    nh_maxamp = 100.0

    num_runs = 50

    pde_solve_maxtime = 10 * 60 * 60
    run_threads = 8
    solver_threads = div(nthreads(), run_threads)

    save_filename = "data2.jld2"

    metadata = (;
        N, S, M, genparams, T, ode_u0, lsks, L, sN,
        sp_epsilon,
        nh_numwaves, nh_baseline, nh_maxamp,
        num_runs,
        pde_solve_maxtime, run_threads,
    )

    ########################################
    # Generate a bunch of params
    ########################################
    params = Vector{BSMMiCRMParams}(undef, num_runs)
    @tasks for ri in 1:num_runs
        params[ri] = gen_many_mms1(N; genparams...)
    end

    ########################################
    # solve ODEs and do linstab
    ########################################
    ode_retcodes = Vector{ReturnCode.T}(undef, num_runs)
    ode_final_states = Vector{Vector{Float64}}(undef, num_runs)
    ode_final_Ts = Vector{Float64}(undef, num_runs)

    linstab_mrls = Vector{Vector{Float64}}(undef, num_runs)

    println("Starting ODE runs")
    flush(stdout)
    @tasks for ri in 1:num_runs
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
    end
    GC.gc()

    ########################################
    # solve PDEs from perturbed ODE solution
    ########################################
    sp_retcodes = Vector{ReturnCode.T}(undef, num_runs)
    sp_final_states = Vector{Matrix{Float64}}(undef, num_runs)
    sp_final_Ts = Vector{Float64}(undef, num_runs)

    println("Starting PDE runs near ODE ss")
    flush(stdout)
    prog1 = Progress(num_runs)
    @tasks for ri in 1:num_runs
        @set ntasks = run_threads

        ode_ss = ode_final_states[ri]
        u0_ = expand_u0_to_size((sN,), ode_ss)
        u0 = perturb_u0_uniform(S, M, u0_, sp_epsilon)

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
    GC.gc()

    ########################################
    # solve PDEs from perturbed ODE solution
    ########################################
    nh_u0s = Vector{Matrix{Float64}}(undef, num_runs)
    nh_retcodes = Vector{ReturnCode.T}(undef, num_runs)
    nh_final_states = Vector{Matrix{Float64}}(undef, num_runs)
    nh_final_Ts = Vector{Float64}(undef, num_runs)

    println("Starting PDE runs from non-homogeneous initial condition")
    flush(stdout)
    prog2 = Progress(num_runs)
    @tasks for ri in 1:num_runs
        @set ntasks = run_threads

        u0 = fill(0.0, S + M, sN)
        for i in 1:S
            xxaa = @view u0[i, :]
            xxaa .= nh_baseline
            add_1d_many_sines2!(xxaa, nh_numwaves, nh_maxamp / nh_maxamp; nmax=10)
        end
        clamp!(u0, 0.0, Inf)
        nh_u0s[ri] = u0

        s = run_1d_pde_sim(params[ri], u0, T, L, sN;
            maxtime=pde_solve_maxtime,
            solver_threads,
        )

        nh_retcodes[ri] = s.retcode
        nh_final_states[ri] = s.u[end]
        nh_final_Ts[ri] = s.t[end]

        s = nothing
        GC.gc()

        next!(prog2)
        flush(stdout)
    end
    GC.gc()

    results = (;
        metadata,
        params,
        ode_retcodes,
        ode_final_states,
        ode_final_Ts,
        linstab_mrls,
        sp_retcodes,
        sp_final_states,
        sp_final_Ts,
        nh_u0s,
        nh_retcodes,
        nh_final_states,
        nh_final_Ts,
    )

    jldsave(save_filename; results...)

    results
end
