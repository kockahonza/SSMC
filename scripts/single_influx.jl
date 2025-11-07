using SSMCMain, SSMCMain.ModifiedMiCRM
using SSMCMain.ModifiedMiCRM.RandomSystems

using Base.Threads, OhMyThreads
using ProgressMeter
using JLD2
using Geppetto
using DataFrames
using DataFramesMeta
using HypothesisTests

################################################################################
# Running single influx scenarios
################################################################################
function do_df_run(Ks, N;
    M=N,
    m=1.0,
    pei=1.0,
    linflux=1.0,
    cinflux=1.0,
    pe=1.0,
    l=0.0,
    c=1.0,
    num_byproducts=M,
    num_repeats=20,
    lsks=10 .^ range(-5, 3, 2000),
    Ds=1e-12, Dr=1.0, Drinflux=Dr,
    print_quality=true,
)
    cms = []
    df = DataFrame(;
        K=Float64[],
        sscode=Int[],
        lscode=Union{Missing,Int}[],
        good_ss=Bool[],
        good_ls=Bool[],
        hss_biomass=Float64[],
        k0mrl=Union{Missing,Float64}[],
        maxmrl=Union{Missing,Float64}[],
        params=Any[],
        steadystates=Vector{Float64}[],
    )
    metadata!(df, "N", N; style=:note)
    metadata!(df, "M", M; style=:note)
    metadata!(df, "m", m; style=:note)
    metadata!(df, "pei", pei; style=:note)
    metadata!(df, "linflux", linflux; style=:note)
    metadata!(df, "cinflux", cinflux; style=:note)
    metadata!(df, "pe", pe; style=:note)
    metadata!(df, "l", l; style=:note)
    metadata!(df, "c", c; style=:note)
    metadata!(df, "num_byproducts", num_byproducts; style=:note)
    metadata!(df, "num_repeats", num_repeats; style=:note)
    metadata!(df, "lsks", lsks; style=:note)
    metadata!(df, "Ds", Ds; style=:note)
    metadata!(df, "Dr", Dr; style=:note)
    metadata!(df, "Drinflux", Drinflux; style=:note)

    @showprogress for K in Ks
        rsg = JansSampler3(N, M;
            K,
            num_influx_resources=1,
            # should be a valid non-dim?
            m,
            # first network layer
            prob_eating_influx=pei,
            linflux, cinflux,
            # rest of the network
            prob_eating=pe,
            l, c,
            # rest
            num_byproducts, # applies to both!
            Ds, Dr, Drinflux
        )

        lst = LinstabScanTester2(
            rsg.Ns + rsg.Nr,
            lsks;
            # zerothr=1000*eps(),
        )
        params, sss, sscodes, lsrslts = example_do_rg_run3(rsg, num_repeats, lst;
            maxresidthr=1e-8,
            tol=1e-13,
            doextinctls=true,
            maxiters=1e6,
            timelimit=30,
        )

        for i in 1:num_repeats
            sscode = sscodes[i]
            good_ss = (sscode in (1, 2))
            lscode = !isnothing(lsrslts[i]) ? lsrslts[i][1] : missing
            good_ls = !ismissing(lscode) ? (lscode in (1, 2)) : true
            push!(df, (
                K, sscode, lscode, good_ss, good_ls,
                sum(sss[i][1:N]),
                !ismissing(lscode) ? lsrslts[i][2] : missing,
                !ismissing(lscode) ? lsrslts[i][3] : missing,
                params[i], sss[i]
            ))
        end

        codes = Int[]
        for i in 1:num_repeats
            if sscodes[i] == 1
                push!(codes, lsrslts[i][1])
            elseif sscodes[i] == 2
                push!(codes, 101)
            else
                push!(codes, sscodes[i])
            end
        end

        cm = countmap(codes)
        push!(cms, cm)
    end
    if print_quality
        @show countmap(df.sscode) countmap(df.lscode) count(df.good_ss .&& df.good_ls) / nrow(df)
        prop_good_ss_and_ls = count(df.good_ss .&& df.good_ls) / nrow(df)
        @show prop_good_ss_and_ls
    end

    df, cms
end
function do_df_run2(Ks, N, B;
    M=N,
    fixed_num_byproducts=false,
    kwargs...
)
    pe = B / M
    num_byproducts = fixed_num_byproducts ? B : Binomial(M, pe)
    do_df_run(Ks, N;
        M,
        pei=pe,
        pe,
        num_byproducts,
        kwargs...
    )
end

function gendata1()
    fname = joinpath("./gd1_" * timestamp() * ".jld2")

    N = 10
    M = N
    B = 3

    # lis = 1.0:-0.02:0.8
    # lis = [1., 0.99, 0.9, 0.8]
    lis = [1.0, 0.9]
    Ks = 10 .^ range(-0.5, 4.0, 10)

    raw_dfs = []
    counts_dfs = []
    for li in lis
        @show li
        flush(stdout)
        df, cms = do_df_run(Ks, N;
            M, pei=1.0,
            linflux=li,
            cinflux=1.0, pe=(B / M),
            l=0.0,
            c=1.0, num_byproducts=B, num_repeats=50,
            lsks=10 .^ range(-5, 3, 2000),
        )
        push!(raw_dfs, df)
        push!(counts_dfs, make_counts_df(df))
    end

    jldsave(fname;
        N, M, B, lis, Ks, raw_dfs, counts_dfs
    )

    fname
end

################################################################################
# Processing data
################################################################################
function make_counts_df(df)
    probsdf = DataFrame(;
        K=Float64[],
        num_runs=Int[],
        bad_ss=Int[],
        extinct=Int[],
        good_ss_bad_ls=Int[],
        nonext_stable=Int[],
        nonext_unstable=Int[],
    )
    for x in groupby(df, :K)
        K = x.K[1]
        num_runs = nrow(x)

        bad_ss = 0
        extinct = 0
        good_ss_bad_ls = 0
        nonext_stable = 0
        nonext_unstable = 0
        for r in eachrow(x)
            if r.sscode == 1
                if r.lscode == 1
                    nonext_stable += 1
                elseif r.lscode == 2
                    nonext_unstable += 1
                else
                    good_ss_bad_ls += 1
                end
            elseif r.sscode == 2
                extinct += 1
            else
                bad_ss += 1
            end
        end
        push!(probsdf, (
            K, num_runs,
            bad_ss,
            extinct,
            good_ss_bad_ls,
            nonext_stable,
            nonext_unstable,
        ))
    end

    probsdf
end

function make_Kli_matrix(f)
    Ks = f["Ks"]
    lis = f["lis"]
    rdfs = f["raw_dfs"]

    rslt = Matrix{@NamedTuple{extinct::Int64, nonext_stable::Int64, nonext_unstable::Int64, bad_ss::Int64, good_ss_bad_ls::Int64, num_runs::Int64}}(undef, length(Ks), length(lis))
    pb = Progress(length(rslt))
    for j in 1:length(lis)
        li = lis[j]
        df = rdfs[j]

        for sdf in groupby(df, :K)
            K = sdf.K[1]
            i = findfirst(==(K), Ks)
            if isnothing(i)
                @show Ks
                throw(ErrorException())
            end

            num_runs = nrow(sdf)

            bad_ss = 0
            extinct = 0
            good_ss_bad_ls = 0
            nonext_stable = 0
            nonext_unstable = 0
            for r in eachrow(sdf)
                if r.sscode == 1
                    if r.lscode == 1
                        nonext_stable += 1
                    elseif r.lscode == 2
                        nonext_unstable += 1
                    else
                        good_ss_bad_ls += 1
                    end
                elseif r.sscode == 2
                    extinct += 1
                else
                    bad_ss += 1
                end
            end
            rslt[i, j] = (;
                extinct,
                nonext_stable,
                nonext_unstable,
                bad_ss,
                good_ss_bad_ls,
                num_runs,
            )

            next!(pb)
        end
    end
    finish!(pb)

    Ks, lis, rslt
end
