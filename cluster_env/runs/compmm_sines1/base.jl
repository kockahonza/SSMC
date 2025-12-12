using Revise
includet("../../../scripts/competing_mms.jl")

import SSMCMain.ModifiedMiCRM.MinimalModelV2

using JLD2

function do_cmm_siny_runs_wrt_K(
    outfname, Ks,
    # Physics params
    m1, c1, l1,
    m2, c2, l2,
    DN1, DN2, DI, DR1, DR2,
    # Run params
    numrepeats, T,
    L, sN,
    # Initial condition params
    meanN0, numwaves, waveampfactor;
    kwargs...
)
    results = Vector{Any}(undef, length(Ks))
    for i in 1:length(Ks)
        K = Ks[i]

        cmmsp = CMMsParams(;
            K=K,
            m1, c1, l1,
            m2, c2, l2,
        )
        Ds = SA[DN1, DN2, DI, DR1, DR2]
        @show cmmsp_get_single_mm_results(cmmsp, Ds...)

        @printf "Starting K=%.3g\n" K
        flush(stdout)

        @time xx = run_siny_cmms(
            cmmsp,
            numrepeats, T,
            L, sN,
            meanN0, numwaves, waveampfactor;
            Ds,
            maxtime=5 * 60,
            run_threads=8,
            save_sols=false,
            kwargs...
        )

        @show countmap(xx.outcomes)
        @printf "Finished K=%.3g\n" K
        flush(stdout)

        results[i] = xx
    end

    params = (;
        Ks, m1, c1, l1, m2, c2, l2, DN1, DN2, DI, DR1, DR2,
        numrepeats, T, L, sN, meanN0, numwaves, waveampfactor,
        outfname, kwargs...
    )

    jldsave(outfname; Ks, results, params)

    results
end

function main1()
    do_cmm_siny_runs_wrt_K(
        "./rslts4_m_c_0.5.jld2", 10 .^ range(0.0, 2.0, 20),
        # Physics params
        0.5, 0.5, 1.0,
        1.0, 1.0, 1.0,
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

function main2()
    do_cmm_siny_runs_wrt_K(
        "./rslts4_m_0.5c_0.9.jld2", 10 .^ range(0.0, 2.0, 20),
        # Physics params
        0.5, 0.9, 1.0,
        1.0, 1.0, 1.0,
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
