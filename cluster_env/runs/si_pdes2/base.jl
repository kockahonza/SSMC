using Revise

includet("../../../scripts/single_influx.jl")

import DataFrames: metadata as md

################################################################################
# New
################################################################################
function kaka(
    N, B, K,
    DN, DI, DR;
    M=N,
    m=base10_lognormal(0.0, 0.001),
    #
    ode_solver=QNDF,
    tol=1e-10,
    distinctthr=0.1,
)
    linflux = 1.0
    l = 0.0

    cinflux = 1.0
    c = 1.0

    pei = pe = B / M
    num_byproducts = Binomial(M, pe)

    rsg = JansSampler3(N, M;
        num_influx_resources=1,
        K, m,
        Ds=DN, Dr=DR, Drinflux=DI,
        l, linflux,
        c, cinflux,
        prob_eating=pe, prob_eating_influx=pei, num_byproducts,
    )

    u0_ = [fill(1.0, N); fill(0.0, M)]

    num_repeats = 20

    ode_solver = QNDF
    tol = 1e-10

    survthresh = 1000 * tol

    sample_params = rsg()
    params = Vector{typeof(sample_params)}(undef, num_repeats)

    retcodes = Vector{SciMLBase.ReturnCode.T}(undef, num_repeats)
    maxresids = Vector{Float64}(undef, num_repeats)

    steadystates = Vector{Vector{Float64}}(undef, num_repeats)
    numsurv = Vector{Int}(undef, num_repeats)

    @tasks for i in 1:num_repeats
        ps = rsg()

        params[i] = ps

        u0 = copy(u0_)
        ssp = make_mmicrm_ss_problem(ps, u0)
        ssps = solve(ssp, DynamicSS(ode_solver());
            abstol=tol,
            reltol=tol,
            callback=make_timer_callback(5),
        )

        retcodes[i] = ssps.retcode
        maxresids[i] = mmicrmmaxresid(ssps.original)

        steadystates[i] = ssps.u
        numsurv[i] = count(ssps.u[1:N]) do x
            x > survthresh
        end


    end

    (; params, retcodes, maxresids, steadystates)
end

function find_distinct_vecs(vecs, threshold)
    if length(vecs) == 0
        eltype(vecs)[]
    end
    uq = eltype(vecs)[vecs[1]]
    counts = Int[1]

    for v in vecs[2:end]
        is_distinct = true
        for (ui, u) in enumerate(uq)
            if all(zip(v, u)) do (x, y)
                abs(x - y) < threshold
            end
                is_distinct = false
                counts[ui] += 1
                break
            end
        end
        if is_distinct
            push!(uq, v)
            push!(counts, 1)
        end
    end

    uq, counts
end

function find_nospace_multistability(
    N, B, K;
    M=N,
    m=base10_lognormal(0.0, 0.001),
    #
    maxN0=100.0,
    num_params=100,
    num_solves=20,
    #
    ode_solver=QNDF,
    tol=1e-10,
    distinctthr=0.1,
)
    linflux = 1.0
    l = 0.0
    cinflux = 1.0
    c = 1.0

    pei = pe = B / M
    num_byproducts = Binomial(M, pe)

    rsg = JansSampler3(N, M;
        num_influx_resources=1,
        K, m,
        l, linflux,
        c, cinflux,
        prob_eating=pe, prob_eating_influx=1., num_byproducts,
    )

    sample_params = rsg()
    params = Vector{typeof(sample_params)}(undef, num_params)

    u0s = Vector{Vector{Vector{Float64}}}(undef, num_params)
    retcodes = Vector{Vector{SciMLBase.ReturnCode.T}}(undef, num_params)
    maxresids = Vector{Vector{Float64}}(undef, num_params)

    steadystates = Vector{Vector{Vector{Float64}}}(undef, num_params)
    numdistsss = Vector{Int}(undef, num_params)

    prog = Progress(num_params)
    @tasks for i in 1:num_params
        ps = rsg()
        params[i] = ps
        u0s[i] = Vector{Vector{Float64}}(undef, num_solves)
        retcodes[i] = Vector{SciMLBase.ReturnCode.T}(undef, num_solves)
        maxresids[i] = Vector{Float64}(undef, num_solves)
        steadystates[i] = Vector{Vector{Float64}}(undef, num_solves)
        for j in 1:num_solves
            u0 = fill(0.0, N + M)
            for k in 1:N
                u0[k] = rand() * maxN0
            end
            u0s[i][j] = copy(u0)

            ssp = make_mmicrm_ss_problem(ps, u0)
            ssps = solve(ssp, DynamicSS(ode_solver());
                abstol=tol,
                reltol=tol,
                callback=make_timer_callback(10),
            )

            retcodes[i][j] = ssps.retcode
            maxresids[i][j] = mmicrmmaxresid(ssps.original)

            steadystates[i][j] = copy(ssps.u)
        end

        uq, _ = find_distinct_vecs(steadystates[i], distinctthr)
        numdistsss[i] = length(uq)

        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)

    (; params, u0s, retcodes, maxresids, steadystates, numdistsss)
end


################################################################################
# Shared/general/util
################################################################################
function run_1d_pde_sim(ps, u0, T, L, sN;
    epsilon=1e-3,
    # solver options
    tol=100000 * eps(),
    maxtime=5 * 60 * 60,
    solver=QNDF,
    solver_threads=nthreads(),
)
    N, M = get_Ns(ps)
    dx = L / sN

    u0_ = if ndims(u0) == 0
        expand_u0_to_size((sN,), [fill(u0, N); fill(0.0, M)])
    elseif ndims(u0) == 1
        expand_u0_to_size((sN,), u0)
    elseif ndims(u0) == 2
        copy(u0)
    else
        throw(ArgumentError("invalid u0 passed to run_1d_pdes_from_df"))
    end
    if !isnothing(epsilon)
        u0_ = perturb_u0_uniform(N, M, u0_, epsilon)
    end

    sps = BSMMiCRMParams(
        ps.mmicrm_params,
        ps.Ds,
        CartesianSpace{1,Tuple{Periodic}}(SA[dx]),
        solver_threads
    )
    sp = make_smmicrm_problem(sps, u0_, T)

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
# First set of functions - runs PDES near steady states
################################################################################
function run_1d_pdes_from_df(fname;
    T=1e6,
    L=5,              # system size in non-dim units
    sN=5000,          # number of spatial points
    epsilon=1e-3,     # perturbation amplitude
    # threading
    run_threads=1,
    total_threads=nthreads(),
    solver_threads=div(total_threads, run_threads),
    # solver options
    maxtime=5 * 60 * 60,
    kwargs...
)
    f = jldopen(fname)
    df = copy(f["df"])
    close(f)

    N = md(df, "N")
    M = md(df, "M")

    metadata!(df, "T", T; style=:note)
    metadata!(df, "L", L; style=:note)
    metadata!(df, "sN", sN; style=:note)
    metadata!(df, "epsilon", epsilon; style=:note)

    num_runs = nrow(df)

    retcodes = Vector{ReturnCode.T}(undef, num_runs)
    final_states = Vector{Matrix{Float64}}(undef, num_runs)
    final_Ts = Vector{Float64}(undef, num_runs)

    prog = Progress(num_runs)
    @tasks for i in 1:num_runs
        @set ntasks = run_threads
        ps = df.params[i]
        ss = df.steadystates[i]

        s = run_1d_pde_sim(ps, ss, T, L, sN;
            epsilon,
            maxtime,
            solver_threads,
            kwargs...
        )

        retcodes[i] = s.retcode
        final_states[i] = s.u[end]
        final_Ts[i] = s.t[end]

        s = nothing
        GC.gc()

        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)

    df.retcodes = retcodes
    df.final_states = final_states
    df.final_Ts = final_Ts

    df
end

"""
Takes a file with a df with params and hss for which it will run a spatial pde starting from a
parturbed ss. Adds the resulting final states and times to the df and saves into a new file
"""
function main1()
    df = run_1d_pdes_from_df("./sel_systems3.jld2";
        run_threads=2,
        solver_threads=64,
        maxtime=5 * 60 * 60,
    )
    jldsave("./rslt_df3.jld2"; df)
end

################################################################################
# Second set of functions - also adds high N0 pde runs to result runs from main1
################################################################################
function add_highN0_run(fname, outfname=nothing;
    prefix="highN0_",
    N0=100.0,
    # threading
    run_threads=1,
    total_threads=nthreads(),
    solver_threads=div(total_threads, run_threads),
    # solver options
    maxtime=5 * 60 * 60,
    kwargs...
)
    f = jldopen(fname)
    df = copy(f["df"])
    close(f)

    metadata!(df, "N0", N0; style=:note)

    T = md(df, "T")
    L = md(df, "L")
    sN = md(df, "sN")
    epsilon = md(df, "epsilon")

    num_runs = nrow(df)

    retcodes = Vector{ReturnCode.T}(undef, num_runs)
    final_states = Vector{Matrix{Float64}}(undef, num_runs)
    final_Ts = Vector{Float64}(undef, num_runs)

    @printf "Starting a run on %s which contains %d rows\n" fname num_runs
    flush(stdout)

    prog = Progress(num_runs)
    @tasks for i in 1:num_runs
        @set ntasks = run_threads
        ps = df.params[i]

        s = run_1d_pde_sim(ps, N0, T, L, sN;
            epsilon,
            maxtime,
            solver_threads,
            kwargs...
        )

        retcodes[i] = s.retcode
        final_states[i] = s.u[end]
        final_Ts[i] = s.t[end]

        s = nothing
        GC.gc()

        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)

    df[:, prefix*"retcodes"] = retcodes
    df[:, prefix*"final_states"] = final_states
    df[:, prefix*"final_Ts"] = final_Ts

    if !isnothing(outfname)
        jldsave(outfname; df)
    end

    df
end

"""
Adds a high N0 run to the same file as would be outputted by main1
"""
function main2()
    df = add_highN0_run("./rslt_df3.jld2", "./rslt2_df3.jld2";
        run_threads=2,
        solver_threads=64,
        maxtime=5 * 60 * 60,
    )
    jldsave("./forsafety.jld2"; df)
end
