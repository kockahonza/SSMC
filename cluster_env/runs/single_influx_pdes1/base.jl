using Revise

includet("../../../scripts/single_influx.jl")

import DataFrames: metadata as md

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
    add_highN0_run("./rslt_df1.jld2", "./rslt2_df1.jld2";
        run_threads=4,
        solver_threads=32,
        maxtime=5 * 60 * 60,
    )
end
