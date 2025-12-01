using Revise
includet("../../../scripts/competing_mms.jl")

import SSMCMain.ModifiedMiCRM.MinimalModelV2

using JLD2

function main1()
    Ks = 10 .^ range(0.0, 2., 20)

    results = Vector{Any}(undef, length(Ks))
    for i in 1:length(Ks)
        K = Ks[i]

        cmmsp = CMMsParams(;
            K=K,
            m1=0.5, c1=0.5, l1=1.0,
            m2=1.0, c2=1.0, l2=1.0,
        )
        Ds = SA[1e-6, 1e-6, 1.0, 1.0, 1.0]
        @show cmmsp_get_single_mm_results(cmmsp, Ds...)

        @printf "Starting K=%.3g\n" K
        flush(stdout)

        @time xx = run_siny_cmms(
            cmmsp,
            50, 1e8,
            10, 5000,
            1.0, 100, 100.0;
            Ds,
            maxtime=5 * 60,
            run_threads=8,
            save_sols=false,
        )

        @show countmap(xx.outcomes)
        @printf "Finished K=%.3g\n" K
        flush(stdout)

        results[i] = xx
    end

    jldsave("./rslts4_m_c_0.5.jld2"; Ks, results)
end
