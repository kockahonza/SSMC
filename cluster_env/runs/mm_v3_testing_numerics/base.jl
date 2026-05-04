using SSMCMain, SSMCMain.ModifiedMiCRM.MinimalModelV2
import SSMCMain.ModifiedMiCRM.MinimalModelV2

using Base.Threads
using OhMyThreads

using ProgressMeter
using JLD2


function main1()
    K = 10.
    l = 0.99
    p = 1.
    DN = 1e-6

    T = 1e8
    L = 10

    u0_base = [100.0, 0.0, 0.0]

    tol=10000 * eps()
    maxtime=60 * 60
    run_threads=4
    solver_threads=div(nthreads(), run_threads)

    dxs = [0.001, 0.01, 0.05, 0.1]

    metadata = (; K, l, p, DN, T, L, u0_base, tol, maxtime, run_threads, solver_threads)

    # Prep bits
    mmp = MMParams(;
        K,
        m=1.,
        c=1.,
        l=l,
        k=0.0,
        d=1.0,
    )

    sols = Vector{Any}(undef, length(dxs))
    @tasks for i in 1:length(dxs)
        @set ntasks = run_threads
        dx = dxs[i]
        sN = Int(L / dx)

        @sprintf "Starting run %d\n" i
        flush(stdout)

        sps = SASMMiCRMParams(
            mmp_to_mmicrm(mmp),
            SA[DN, 1., p],
            make_cartesianspace_smart(1; dx),
            solver_threads,
        )
        u0 = perturb_u0_uniform(1, 2, expand_u0_to_size((sN,), u0_base), 1e-3)
        sp = make_smmicrm_problem(sps, u0, T)

        s = solve(sp, QNDF();
            dense=false,
            save_everystep=false,
            abstol=tol,
            reltol=tol,
            callback=make_timer_callback(maxtime),
        )

        sols[i] = s

        @sprintf "Finished run %d\n" i
        flush(stdout)
    end

    jldsave("data1.jld2"; dxs, sols, metadata)
end
