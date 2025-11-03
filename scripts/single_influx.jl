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

    @showprogress for K in Ks
        rsg = JansSampler3(N, M;
            K,
            num_influx_resources=1,
            # should be a valid non-dim?
            m=1.0,
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

    @show countmap(df.sscode) countmap(df.lscode) count(df.good_ss .&& df.good_ls) / nrow(df)
    prop_good_ss_and_ls = count(df.good_ss .&& df.good_ls) / nrow(df)
    @show prop_good_ss_and_ls

    df, cms
end
function do_df_run2(Ks, N, B; M=N, kwargs...)
    pe = B / M
    do_df_run(Ks, N;
        M,
        pei=pe,
        pe,
        num_byproducts=B,
        kwargs...
    )
end

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
# Plotting
################################################################################
function plot_binom_sample!(ax, xs, ns, num_repeats;
    label="",
    proportions=false,
)
    if isa(num_repeats, Number)
        num_repeats = fill(num_repeats, length(ns))
    end

    xx = if proportions
        ns ./ num_repeats
    else
        ns
    end
    sl = scatterlines!(ax, xs, xx; label)

    mins = Float64[]
    maxs = Float64[]
    @show proportions
    for (n, nrs) in zip(ns, num_repeats)
        bt = BinomialTest(n, nrs)
        ci = confint(bt; method=:wilson)
        if proportions
            push!(mins, ci[1])
            push!(maxs, ci[2])
        else
            push!(mins, ci[1] * nrs)
            push!(maxs, ci[2] * nrs)
        end
    end

    b = band!(xs, mins, maxs;
        alpha=0.5
    )

    (sl, b)
end

function plot_cdf1!(ax, cdf; include_bad=true, proportions=false)
    goodruns = cdf.num_runs .- cdf.bad_ss .- cdf.good_ss_bad_ls
    p1 = plot_binom_sample!(ax, cdf.K, cdf.extinct, goodruns;
        label="extinct",
        proportions
    )
    # lsruns = cdf.num_runs .- cdf.bad_ss .- cdf.extinct
    p2 = plot_binom_sample!(ax, cdf.K, cdf.nonext_unstable, goodruns;
        label="unstable",
        proportions
    )
    p3 = plot_binom_sample!(ax, cdf.K, cdf.nonext_stable, goodruns;
        label="stable",
        proportions
    )
    if include_bad
        p3 = plot_binom_sample!(ax, cdf.K, cdf.bad_ss .+ cdf.good_ss_bad_ls, cdf.num_runs;
            label="bad data",
            proportions
        )
    end
    (p1, p2, p3)
end
function plot_cdf1(cdf; kwargs...)
    f = Figure()
    ax = Axis(f[1, 1];
        xlabel="K",
        xscale=log10,
    )
    p = plot_cdf1!(ax, cdf; kwargs...)
    axislegend(ax)
    FigureAxisAnything(f, ax, p)
end

function plot_df1(df)
    f = Figure(;
    # size=(800, 500)
    )

    ax_biom = Axis(f[1, 1:2];
        ylabel="HSS biomass",
        yscale=Makie.pseudolog10,
        # yscale=Makie.log10,
        xscale=log10,
    )
    ax2 = Axis(f[2, :];
        ylabel="observed MRL",
        xscale=log10,
    )
    ax3 = Axis(f[3, :];
        ylabel="MRL at K=0",
        xscale=log10,
    )
    ax4 = Axis(f[4, 1];
        ylabel="Counts",
        xscale=log10,
    )
    linkxaxes!(ax_biom, ax4)
    linkxaxes!(ax2, ax4)
    linkxaxes!(ax2, ax4)
    for ax in [ax_biom, ax2, ax3]
        hidexdecorations!(ax;
            grid=false,
            ticks=false,
        )
    end
    ax4.xlabel = "K"

    xx = @subset df :good_ss
    # scatter!(ax_biom, xx.K, clamp.(xx.hss_biomass, 1e-20, Inf))
    scatter!(ax_biom, xx.K, xx.hss_biomass)

    scatter!(ax2, xx.K, xx.maxmrl)
    maxmaxmrl = maximum(xx.maxmrl)
    ylims!(ax2, (-1.1, 1.1) .* abs(maxmaxmrl))
    text!(ax2, 0.05, 0.75;
        text=(@sprintf "max is %.5g" maxmaxmrl),
        space=:relative,
    )

    scatter!(ax3, xx.K, xx.k0mrl)
    maxk0mrl = maximum(xx.k0mrl)
    ylims!(ax3, (-1.1, 1.1) .* abs(maxk0mrl))
    text!(ax3, 0.05, 0.75;
        text=(@sprintf "max is %.5g" maxk0mrl),
        space=:relative,
    )

    # display(GLMakie.Screen(), f)

    rowgap!(f.layout, 1.0)
    plot_cdf1!(ax4, make_counts_df(df))
    # axislegend(ax4; position=:lc)
    Legend(f[4, 2], ax4; padding=0.0)

    f
end
