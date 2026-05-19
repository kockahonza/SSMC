using Revise

using SSMCMain, SSMCMain.ModifiedMiCRM
using SSMCMain.ModifiedMiCRM.TwoMMs

using Base.Threads
using OhMyThreads
using ProgressMeter

using JLD2

function run_siny_tmms(
    mmicrcm_params::AbstractMMiCRMParams, numrepeats, T,
    L, sN,
    meanN0, numwaves, waveampfactor,
    ;
    Ds=SA[1e-6, 1e-6, 1.0, 1.0, 1.0],
    tol=1e-8,
    extthreshold=100 * tol,
    maxtime=60,
    save_sols=true,
    # threading
    run_threads=1,
    total_threads=nthreads(),
    solver_threads=div(total_threads, run_threads),
)
    dx = L / sN

    retcodes = Vector{ReturnCode.T}(undef, numrepeats)
    u0s = Vector{Matrix{Float64}}(undef, numrepeats)
    fss = Vector{Matrix{Float64}}(undef, numrepeats)
    fTs = Vector{Float64}(undef, numrepeats)
    sols = Vector{Any}(undef, numrepeats)

    prog = Progress(numrepeats)
    @tasks for i in 1:numrepeats
        @set ntasks = run_threads
        u0 = get_siny_u0(2, 3, sN, dx, meanN0, numwaves, waveampfactor)

        sps = BSMMiCRMParams(
            mmicrcm_params,
            Ds,
            CartesianSpace{1,Tuple{Periodic}}(SA[dx]),
            solver_threads
        )

        p = make_smmicrm_problem(sps, u0, T)
        s = solve(p, QNDF();
            dense=false,
            save_everystep=save_sols,
            abstol=tol, reltol=tol,
            callback=make_timer_callback(maxtime),
        )

        retcodes[i] = s.retcode
        u0s[i] = u0
        fss[i] = s.u[end]
        fTs[i] = s.t[end]
        sols[i] = save_sols ? s : nothing

        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)

    (; retcodes, fss, fTs, u0s, sols)
end

function do_tmm_siny_runs_wrt_K(
    outfname, Ks,
    # Physics params
    m1, c1, l1, k1,
    m2, c2, l2, k2,
    DN1, DN2, DI, DR1, DR2,
    # Run params
    numrepeats, T,
    L, sN,
    # Initial condition params
    meanN0, numwaves, waveampfactor;
    kwargs...
)
    Ds = SA[DN1, DN2, DI, DR1, DR2]

    tmm_params = Vector{Any}(undef, length(Ks))
    results = Vector{Any}(undef, length(Ks))
    for i in 1:length(Ks)
        K = Ks[i]

        tmmps = TMMsParams(;
            K=K,
            m1, c1, l1, k1,
            m2, c2, l2, k2,
        )

        @printf "Starting K=%.3g\n" K
        flush(stdout)

        @time xx = run_siny_tmms(
            tmmsp_to_mmicrm(tmmps),
            numrepeats, T,
            L, sN,
            meanN0, numwaves, waveampfactor;
            Ds,
            maxtime=5 * 60,
            run_threads=8,
            save_sols=false,
            kwargs...
        )

        @printf "Finished K=%.3g\n" K
        flush(stdout)

        tmm_params[i] = tmmps
        results[i] = xx
    end

    metadata = (;
        Ks, m1, c1, l1, m2, c2, l2, DN1, DN2, DI, DR1, DR2,
        numrepeats, T, L, sN, meanN0, numwaves, waveampfactor,
        outfname, kwargs...
    )

    jldsave(outfname; Ks, tmm_params, results, metadata)

    results
end

################################################################################
# Changing k
################################################################################
function main1()
    for k in [0.0, 0.1, 0.4, 0.5, 0.6, 0.9, 1.0]
        outfname = "./d1_k$(k).jld2"
        do_tmm_siny_runs_wrt_K(
            outfname, 10 .^ range(0.4, 1.7, 15),
            # Physics params
            0.9, 0.9, 1.0, k,
            1.0, 1.0, 1.0, k,
            1e-6, 1e-6, 1.0, 1.0, 1.0,
            # Run params
            50, 1e8,
            10, 5000,
            # Initial condition params
            1.0, 100, 100.0;
            maxtime=5 * 60,
            run_threads=8,
            save_sols=false,
        )
    end
end

"""
Zooming in on smaller ks. Reduced the number of random runs as they all seem the same.
"""
function main3_ks2()
    for k in [0.0, 0.001, 0.01, 0.1, 0.2]
        outfname = "./wrtk_2/d1_k$(k).jld2"
        do_tmm_siny_runs_wrt_K(
            outfname, 10 .^ range(0.3, 2., 20),
            # Physics params
            0.9, 0.9, 1.0, k,
            1.0, 1.0, 1.0, k,
            1e-6, 1e-6, 1.0, 1.0, 1.0,
            # Run params
            20, 1e8,
            10, 5000,
            # Initial condition params
            1.0, 100, 100.0;
            maxtime=5 * 60,
            run_threads=8,
            save_sols=false,
        )
    end
end

"""
Same as above but even more
"""
function main4_ks3()
    for k in round.(10 .^ range(-2, log10(0.15), 5); digits=4)
        outfname = "./wrtk_3/d1_k$(k).jld2"
        do_tmm_siny_runs_wrt_K(
            outfname, 10 .^ range(0.3, 2., 20),
            # Physics params
            0.9, 0.9, 1.0, k,
            1.0, 1.0, 1.0, k,
            1e-6, 1e-6, 1.0, 1.0, 1.0,
            # Run params
            20, 1e8,
            10, 5000,
            # Initial condition params
            1.0, 100, 100.0;
            maxtime=5 * 60,
            run_threads=8,
            save_sols=false,
        )
    end
end

"""
Same as above but even more
"""
function main5_ks4()
    for k in [0.08, 0.09, 0.1, 0.11, 0.12]
        outfname = "./wrtk_4/d1_k$(k).jld2"
        do_tmm_siny_runs_wrt_K(
            outfname, 10 .^ range(0.3, 2., 20),
            # Physics params
            0.9, 0.9, 1.0, k,
            1.0, 1.0, 1.0, k,
            1e-6, 1e-6, 1.0, 1.0, 1.0,
            # Run params
            20, 1e8,
            10, 5000,
            # Initial condition params
            1.0, 100, 100.0;
            maxtime=5 * 60,
            run_threads=8,
            save_sols=false,
        )
    end
end

################################################################################
# Changing p/D_R
################################################################################
function main2_ps()
    for p in [1., 0.5, 0.1, 0.01]
        outfname = "./wrtp/d1_p$(p).jld2"
        do_tmm_siny_runs_wrt_K(
            outfname, 10 .^ range(0.3, 2.0, 20),
            # Physics params
            0.9, 0.9, 1.0, 0.0,
            1.0, 1.0, 1.0, 0.0,
            1e-6, 1e-6, 1.0, p, p,
            # Run params
            25, 1e8,
            10, 5000,
            # Initial condition params
            1.0, 100, 100.0;
            maxtime=5 * 60,
            run_threads=8,
            save_sols=false,
        )
    end
end

function main6_ps2()
    for p in [1., 0.5, 0.1, 0.01]
        outfname = "./wrtp_2/d1_p$(p).jld2"
        do_tmm_siny_runs_wrt_K(
            outfname, 10 .^ range(0.3, 2.0, 20),
            # Physics params
            0.9, 0.9, 1.0, 0.0,
            1.0, 1.0, 1.0, 0.0,
            1e-6, 1e-6, 1.0, p, p,
            # Run params
            25, 1e8,
            10, 5000,
            # Initial condition params
            1.0, 100, 100.0;
            maxtime=8 * 60,
            run_threads=8,
            save_sols=false,
        )
    end
end
