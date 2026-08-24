include("../../../scripts/single_influx.jl")

################################################################################
# Same kind of run as cluster_env/runs/single_influx, but the random systems
# come from get_si_sampler_for_paper rather than an inline JansSampler3.
#
# The output layout (Ks, lis, raw_dfs and the raw_df columns) is kept identical
# to single_influx so make_counts_df/make_Kli_matrix/make_Kli_matrix_raw all
# still work unchanged.
################################################################################

"""
Mirror of `do_df_run` from scripts/single_influx.jl, but each K uses
`get_si_sampler_for_paper(K, li, DN; N, M, B, DR, ...)` to generate systems.

Any extra kwargs (m, c, cinflux, ...) are forwarded straight to
get_si_sampler_for_paper, so its defaults apply unless overridden here.
"""
function do_df_run_paper(Ks, li;
    DN=0.0,
    N=20, M=N, B=3,
    DR=1.0,
    num_repeats=20,
    lsks=10 .^ range(-5, 3, 2000),
    print_quality=true,
    sampler_kwargs...
)
    cms = []
    rsgs = []
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
    metadata!(df, "B", B; style=:note)
    metadata!(df, "linflux", li; style=:note)
    metadata!(df, "DN", DN; style=:note)
    metadata!(df, "DR", DR; style=:note)
    metadata!(df, "num_repeats", num_repeats; style=:note)
    metadata!(df, "lsks", lsks; style=:note)
    metadata!(df, "sampler_kwargs", values(sampler_kwargs); style=:note)

    @showprogress for K in Ks
        rsg = get_si_sampler_for_paper(K, li, DN; N, M, B, DR, sampler_kwargs...)
        push!(rsgs, rsg)

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

        push!(cms, countmap(codes))
    end
    if print_quality
        @show countmap(df.sscode) countmap(df.lscode)
        prop_good_ss_and_ls = count(df.good_ss .&& df.good_ls) / nrow(df)
        @show prop_good_ss_and_ls
    end

    df, cms, rsgs
end

"""
Equivalent of gendata5() in cluster_env/runs/single_influx/base.jl - a scan over
(K, linflux) - but with the systems drawn from get_si_sampler_for_paper.
"""
function gendata1(; DN=0.0, DR=1.0)
    fname = joinpath("./gd1_" * timestamp() * ".jld2")

    N = 20
    M = N
    B = 3

    xx = LeakageScale.ltox(0.999)
    xxs = range(0.0, xx, 30)
    lis = LeakageScale.l.(xxs)

    Ks = 10 .^ range(-0.5, 4.0, 50)

    num_repeats = 100
    lsks = 10 .^ range(-5, 3, 2000)

    raw_dfs = []
    all_rsgs = []
    for li in lis
        @show li
        flush(stdout)
        df, cms, rsgs = do_df_run_paper(Ks, li;
            DN, DR,
            N, M, B,
            num_repeats,
            lsks,
        )
        push!(raw_dfs, df)
        push!(all_rsgs, rsgs)
        flush(stdout)
    end

    jldsave(fname;
        N, M, B, DN, DR, num_repeats, lsks,
        lis, Ks, raw_dfs, all_rsgs
    )

    fname
end

"""
The same as gendata1 but B=5 to see how this affects things.
"""
function gendata2(; DN=0.0, DR=1.0)
    dname = joinpath("./gd2_B5_" * timestamp())
    mkpath(dname)

    N = 20
    M = N
    B = 5

    xx = LeakageScale.ltox(0.999)
    xxs = range(0.0, xx, 30)
    lis = LeakageScale.l.(xxs)

    Ks = 10 .^ range(-0.5, 4.0, 50)

    num_repeats = 100
    lsks = 10 .^ range(-5, 3, 2000)

    rowname(i) = joinpath(dname, "row" * lpad(i, 4, '0') * ".jld2")

    for (i, li) in enumerate(lis)
        @show li
        flush(stdout)
        df, cms, rsgs = do_df_run_paper(Ks, li;
            DN, DR,
            N, M, B,
            num_repeats,
            lsks,
        )
        jldsave(rowname(i); li, df, rsgs)
        flush(stdout)
    end

    # Best-effort merge of the per-li rows back into the single-file layout the
    # rest of the analysis code expects. This holds everything in memory at once,
    # which is what the 94G cap could not take before, so it is allowed to fail -
    # the rows above are the real output and the merge can be redone anywhere.
    fname = dname * ".jld2"
    try
        raw_dfs = []
        all_rsgs = []
        for i in eachindex(lis)
            row = JLD2.load(rowname(i))
            push!(raw_dfs, row["df"])
            push!(all_rsgs, row["rsgs"])
        end
        jldsave(fname;
            N, M, B, DN, DR, num_repeats, lsks,
            lis, Ks, raw_dfs, all_rsgs
        )
        @info "merged rows into $fname"
    catch e
        @error "merge failed, per-li rows in $dname are intact" exception = (e, catch_backtrace())
    end

    dname
end
