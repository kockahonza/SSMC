using Revise
includet("../../../scripts/competing_mms.jl")

import SSMCMain.ModifiedMiCRM.MinimalModelV2

using JLD2

function main1()
    Ks = 10 .^ range(0.0, 3.0, 20)

    results = Vector{Any}(undef, length(Ks))
    for i in 1:length(Ks)
        K = Ks[i]

        cmmsp = CMMsParams(;
            K=K,
            m1=0.9, c1=0.9, l1=1.0,
            m2=1.0, c2=1.0, l2=1.0,
        )
        Ds = SA[1e-6, 1e-6, 1.0, 1.0, 1.0]
        @show cmmsp_get_single_mm_results(cmmsp, Ds...)

        @printf "Starting K=%.3g" K
        flush(stdout)

        @time xx = run_siny_cmms(
            cmmsp,
            50, 1e8,
            10, 10000,
            1.0, 100, 20.0;
            Ds,
            maxtime=60 * 60,
            run_threads=4,
            save_sols=false,
        )

        @show countmap(xx.outcomes)
        @printf "Finished K=%.3g" K
        flush(stdout)

        results[i] = xx
    end

    jldsave("./rslts2.jld2"; Ks, results)
end


