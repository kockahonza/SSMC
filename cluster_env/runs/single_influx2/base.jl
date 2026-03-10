using SSMCMain, SSMCMain.ModifiedMiCRM
using SSMCMain.ModifiedMiCRM.RandomSystems

using Base.Threads, OhMyThreads
using ProgressMeter
using JLD2
using Geppetto

function main1()
    N = 20
    M = N
    B = 3

    linflux = 1.0

    num_repeats = 10

    lsks = 10 .^ range(-5, 3, 2000)
    lst = LinstabScanTester2(
        N + M,
        lsks;
        # zerothr=1000*eps(),
    )

    metadata = (;
        N, M, B, linflux, num_repeats, lsks
    )

    fname = joinpath("./m1" * timestamp() * ".jld2")

    Kmcrs = []

    for K in 10 .^ range(-0.5, 4.0, 2)
        m = base10_lognormal(0.0, 0.001)
        c = base10_lognormal(0.0, 0.001)
        push!(Kmcrs, (K, m, c, 1.0))
    end
    for m_ in range(-4.0, 0.5, 2)
        m = base10_lognormal(m_, 0.001)
        c = base10_lognormal(0.0, 0.001)
        push!(Kmcrs, (1.0, m, c, 1.0))
    end
    for c_ in range(-0.5, 4.0, 2)
        m = base10_lognormal(0.0, 0.001)
        c = base10_lognormal(c_, 0.001)
        push!(Kmcrs, (1.0, m, c, 1.0))
    end
    for r in 10 .^ range(-4.0, 0.5, 2)
        m = base10_lognormal(0.0, 0.001)
        c = base10_lognormal(0.0, 0.001)
        push!(Kmcrs, (1.0, m, c, r))
    end

    rsgs = [
        JansSampler3(N, M;
            K, m, r,
            cinflux=c, c,
            linflux, l=0.0,
            num_influx_resources=1,
            prob_eating_influx=1.0,
            prob_eating=B / M,
            num_byproducts=B,
            Ds=0.0, Dr=1.0, Drinflux=1.0
        ) for (K, m, c, r) in Kmcrs
    ]

    results = []

    prog = Progress(length(rsgs))
    for rsg in rsgs
        params, sss, sscodes, lsrslts = example_do_rg_run3(rsg, num_repeats, lst;
            maxresidthr=1e-8,
            tol=1e-13,
            doextinctls=true,
            maxiters=1e6,
            timelimit=30,
        )

        bad_ss = 0
        extinct = 0
        good_ss_bad_ls = 0
        nonext_stable = 0
        nonext_unstable = 0
        for i in 1:num_repeats
            lscode = !isnothing(lsrslts[i]) ? lsrslts[i][1] : missing
            if sscodes[i] == 1
                if lscode == 1
                    nonext_stable += 1
                elseif lscode == 2
                    nonext_unstable += 1
                else
                    good_ss_bad_ls += 1
                end
            elseif sscodes[i] == 2
                extinct += 1
            else
                bad_ss += 1
            end
        end

        push!(results, (; extinct, nonext_stable, nonext_unstable, bad_ss, good_ss_bad_ls))

        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)

    jldsave(fname;
        metadata, Kmcrs, rsgs, results
    )
    (Kmcrs, results)
end
