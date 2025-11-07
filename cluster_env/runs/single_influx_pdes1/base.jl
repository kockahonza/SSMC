using Revise

includet("../../../scripts/single_influx.jl")

import DataFrames: metadata as md

function run_1d_pdes_from_df(fname;
    T=1e6,
    L=5,              # system size in non-dim units
    sN=5000,          # number of spatial points
    epsilon=1e-3,     # perturbation amplitude
    # solver options
    tol=100000 * eps(),
    maxtime=5 * 60 * 60,
    solver=QNDF,
    # threading
    run_threads=1,
    total_threads=nthreads(),
    solver_threads=div(total_threads, run_threads),
)
    dx = L / sN

    f = jldopen(fname)
    df = copy(f["df"])
    close(f)

    N = md(df, "N")
    M = md(df, "M")

    num_runs = nrow(df)

    retcodes = Vector{ReturnCode.T}(undef, num_runs)
    final_states = Vector{Matrix{Float64}}(undef, num_runs)
    final_Ts = Vector{Float64}(undef, num_runs)

    prog = Progress(num_runs)

    @tasks for i in 1:num_runs
        @set ntasks = run_threads
        ps = df.params[i]
        ss = df.steadystates[i]

        u0 = expand_u0_to_size((sN,), ss)
        pu0 = perturb_u0_uniform(N, M, u0, epsilon)

        sps = BSMMiCRMParams(
            ps.mmicrm_params,
            ps.Ds,
            CartesianSpace{1,Tuple{Periodic}}(SA[dx]),
            solver_threads
        )
        sp = make_smmicrm_problem(sps, pu0, T)

        s = solve(sp, solver();
            dense=false,
            save_everystep=false,
            abstol=tol,
            reltol=tol,
            callback=make_timer_callback(maxtime)
        )

        retcodes[i] = s.retcode
        final_states[i] = s.u[end]
        final_Ts[i] = s.t[end]

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
