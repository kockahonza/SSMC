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
function binary_random_matrix(a, b, p)
    # Generate a random binary matrix of size a x b with probability of 1 = p
    r = rand(a, b)
    m = fill(0, a, b)

    for i in 1:a
        for j in 1:b
            if r[i, j] < p
                m[i, j] = 1
            end
        end
    end

    return m
end

function marsland_initialization(S, M;
    SA=5, MA=5, q=0.9, c0=0.0, c1=1.0, muc=10,
    fs=0.45, fw=0.45, sparsity=0.2
)
    @assert S > 5 && M > 5 "S and M must be at least 6 each"

    # let us begin by assigning the whole array c0
    c = fill(c0 / M, (S, M))

    # now we will calculate the block structure of the matrix
    F = ceil(M / MA) #number of resource classes
    T = ceil(S / SA) #number of species classes
    S_overlap = S % SA # number of species in the last class
    M_overlap = M % MA # number of resources in the last class
    # we will always assume that the last species class is the "general" class 
    # println("T: ", T, " F: ", F)

    # we will sample the consumption matrix in block form

    for tt in 1:T
        for ff in 1:F
            if tt != T
                if ff == tt
                    p = muc / (M * c1) * (1 + q * (M - MA) / M)
                else
                    p = muc / (M * c1) * (1 - q)
                end

                if ff * MA > M
                    # ensure that the last block is not larger than the matrix
                    block = binary_random_matrix(SA, M_overlap, p)
                    c[Int(1 + SA * (tt - 1)):Int(tt * SA), Int(1 + MA * (ff - 1)):Int(M)] .= block
                else
                    block = binary_random_matrix(SA, MA, p)
                    c[Int(1 + SA * (tt - 1)):Int(tt * SA), Int(1 + MA * (ff - 1)):Int(ff * MA)] .= block
                end

            else
                # generalist class
                p = muc / (M * c1)
                if S_overlap != 0
                    block = binary_random_matrix(S_overlap, M, p)
                    c[Int(1 + SA * (tt - 1)):Int(S), :] .= block
                else

                    block = binary_random_matrix(SA, M, p)
                    c[Int(1 + SA * (tt - 1)):Int(tt * SA), :] .= block
                end
            end
        end
    end

    # Time for D_iab
    strain_class = 0
    D = fill(0.0, (S, M, M))
    for i in 1:S
        if (i - 1) % SA == 0
            strain_class += 1
            # println("Class: ", class, " i: ", i)
        end

        resource_class = 0
        for j in 1:M
            if (j - 1) % MA == 0
                resource_class += 1
            end

            #start with background levels
            bkg = (1 - fw - fs) / (M - MA)
            p = fill(bkg, M)

            if resource_class == strain_class
                if strain_class == T
                    if M_overlap != 0
                        p[(M-M_overlap):M] .= fw + fs
                    else
                        p[(M-MA):M] .= fw + fs
                    end
                else
                    #the within class resource
                    upper_limit = minimum(((strain_class - 1) * MA + MA, M))
                    p[1+(strain_class-1)*MA:upper_limit] .= fs

                    # the waste resources
                    if M_overlap != 0
                        p[(M-M_overlap):M] .= fw
                    else
                        p[(M-MA):M] .= fw
                    end
                end
            else
                p = fill(1.0, M)
            end



            #lets sample the distribution
            vec = rand(Dirichlet(p))
            D[i, :, j] = vec

        end
    end

    # constant dilution rate
    rnd = rand()
    r = fill(rnd, M)

    # universal death rate
    rnd2 = rand()
    m = fill(rnd2, S)

    # for simplicity, lets start with a single fed resource
    # chemostat feed rate 
    #K = fill(0.,M)
    #K[1] = 1.

    # lets allow resources some variability
    #K_dist = truncated(Normal(0.5,0.1), 0.0, 1.0)
    K_dist = Beta(0.1, 0.3)
    K = rand(K_dist, M)
    # K ./= 2.

    # leakage now. Lets assume its a pretty flat probability distribution
    leak = Beta(0.2 / sparsity, 0.2)
    l = rand(leak, (S, M))

    Ds = fill(0.0, (S + M))
    Ds[1:S] .= 1e-6
    Ds[(S+1):(S+M)] .= 1e-3
    for a in 1:M
        if K[a] > 0.5
            Ds[S+a] = 1.0
        end
    end
    # Ds[1+S] = 1
    # Ds[S+2:S+M] .= 0.1

    ps = BMMiCRMParams(fill(1.0, S), fill(1.0, M), m, K, r, l, c, D)
    BSMMiCRMParams(ps, Ds)
end

################################################################################
# Main functions
################################################################################
function testing1()
    S = 10
    M = 10

    T = 1000000000

    ode_u0 = [fill(1.0, S); fill(0.0, M)]

    L = 1
    sN = 500

    sp_epsilon = 1e-3

    nh_numwaves = 10
    nh_baseline = 10.0
    nh_maxamp = 5.0

    num_runs = 3

    pde_solve_maxtime = 1 * 60
    run_threads = 4
    solver_threads = div(nthreads(), run_threads)

    save_filename = "data1.jld2"

    metadata = (;
        S, M, ode_u0, T,
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
        params[ri] = marsland_initialization(S, M)
    end

    ########################################
    # solve ODEs
    ########################################
    ode_retcodes = Vector{ReturnCode.T}(undef, num_runs)
    ode_final_states = Vector{Vector{Float64}}(undef, num_runs)
    ode_final_Ts = Vector{Float64}(undef, num_runs)

    println("Starting ODE runs")
    flush(stdout)
    @tasks for ri in 1:num_runs
        # @set ntasks = run_threads
        p = make_mmicrm_problem(params[ri], copy(ode_u0), T)
        s = solve(p)
        ode_retcodes[ri] = s.retcode
        ode_final_states[ri] = s.u[end]
        ode_final_Ts[ri] = s.t[end]
    end

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
            N = @view u0[i, :]
            N .= nh_baseline
            add_1d_many_sines2!(N, nh_numwaves, nh_maxamp / nh_maxamp; nmax=10)
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

    results = (;
        metadata,
        params,
        ode_retcodes,
        ode_final_states,
        ode_final_Ts,
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

function main1()
    S = 30
    M = 30

    T = 1000000000

    ode_u0 = [fill(1.0, S); fill(0.0, M)]

    L = 10
    sN = 5000

    sp_epsilon = 1e-3

    nh_baseline = 10.0
    nh_numwaves = 10
    nh_maxamp = 10.0

    num_runs = 100

    pde_solve_maxtime = 60 * 60
    run_threads = 8
    solver_threads = div(nthreads(), run_threads)

    save_filename = "data1.jld2"

    metadata = (;
        S, M, ode_u0, T, L, sN,
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
        params[ri] = marsland_initialization(S, M)
    end

    ########################################
    # solve ODEs
    ########################################
    ode_retcodes = Vector{ReturnCode.T}(undef, num_runs)
    ode_final_states = Vector{Vector{Float64}}(undef, num_runs)
    ode_final_Ts = Vector{Float64}(undef, num_runs)

    println("Starting ODE runs")
    flush(stdout)
    @tasks for ri in 1:num_runs
        # @set ntasks = run_threads
        p = make_mmicrm_problem(params[ri], copy(ode_u0), T)
        s = solve(p)
        ode_retcodes[ri] = s.retcode
        ode_final_states[ri] = s.u[end]
        ode_final_Ts[ri] = s.t[end]
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
            N = @view u0[i, :]
            N .= nh_baseline
            add_1d_many_sines2!(N, nh_numwaves, nh_maxamp / nh_maxamp; nmax=10)
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

function main2()
    S = 30
    M = 30

    T = 1000000000

    ode_u0 = [fill(1.0, S); fill(0.0, M)]

    L = 5
    sN = 5000

    sp_epsilon = 1e-3

    nh_baseline = 10.0
    nh_numwaves = 10
    nh_maxamp = 10.0

    num_runs = 100

    pde_solve_maxtime = 5 * 60 * 60
    run_threads = 8
    solver_threads = div(nthreads(), run_threads)

    save_filename = "data2.jld2"

    metadata = (;
        S, M, ode_u0, T, L, sN,
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
        params[ri] = marsland_initialization(S, M)
    end

    ########################################
    # solve ODEs
    ########################################
    ode_retcodes = Vector{ReturnCode.T}(undef, num_runs)
    ode_final_states = Vector{Vector{Float64}}(undef, num_runs)
    ode_final_Ts = Vector{Float64}(undef, num_runs)

    println("Starting ODE runs")
    flush(stdout)
    @tasks for ri in 1:num_runs
        # @set ntasks = run_threads
        p = make_mmicrm_problem(params[ri], copy(ode_u0), T)
        s = solve(p)
        ode_retcodes[ri] = s.retcode
        ode_final_states[ri] = s.u[end]
        ode_final_Ts[ri] = s.t[end]
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
            N = @view u0[i, :]
            N .= nh_baseline
            add_1d_many_sines2!(N, nh_numwaves, nh_maxamp / nh_maxamp; nmax=10)
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
