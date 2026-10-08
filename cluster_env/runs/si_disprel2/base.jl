using Revise
using SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.MinimalModelV3

include("../../../scripts/single_influx.jl")

using Base.Threads, OhMyThreads
using Printf
using ProgressMeter
using JLD2, Geppetto
using Random, Distributions
using DataFrames, DataFramesMeta
using Optim

################################################################################
# Cluster stage: generate SI systems and find their well-mixed steady states
################################################################################
function solve_si_odes(
    outfname, num_runs,
    K, l, p,
    T, tol;
    abstol=100*tol,
    reltol=tol,
    DN=0.,
    N=20, M=20,
    si_u0=[fill(1., N); fill(0., M)],
    solver=TRBDF2,
    maxtime=30.,
    extinction_threshold=abstol,
)
    rsg = get_si_sampler_for_paper(K, l, DN; DR=p, N, M)
    N = rsg.Ns
    metadata = (;
        num_runs, K, l, p, T, abstol, reltol, DN, N, M, si_u0, solver, maxtime,
        extinction_threshold,
        rsg,
    )

    params = Vector{BMMiCRMParams}(undef, num_runs)
    Dss = Vector{Vector{Float64}}(undef, num_runs)
    retcodes = Vector{ReturnCode.T}(undef, num_runs)
    final_states = Vector{Vector{Float64}}(undef, num_runs)
    final_Ts = Vector{Float64}(undef, num_runs)
    maxresids = Vector{Float64}(undef, num_runs)
    num_surv = Vector{Int}(undef, num_runs)

    prog = Progress(num_runs)
    @tasks for i in 1:num_runs
        gen_ps = rsg()
        si_ps = gen_ps.mmicrm_params
        si_Ds = gen_ps.Ds

        si_p = make_mmicrm_problem(si_ps, copy(si_u0), T)
        si_s = solve(si_p, solver();
            dense=false,
            save_everystep=false,
            callback=CallbackSet(make_timer_callback(maxtime), make_ode_extinction_exit_callback(N, extinction_threshold), PositiveDomain(copy(si_u0); save=false)),
            abstol=abstol,
            reltol=reltol,
        )
        si_fs = si_s.u[end]

        params[i] = si_ps
        Dss[i] = si_Ds
        retcodes[i] = si_s.retcode
        final_states[i] = si_fs
        final_Ts[i] = si_s.t[end]
        maxresids[i] = mmicrmmaxresid(si_s)
        num_surv[i] = count(>(extinction_threshold), si_fs[1:N])

        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)

    df = DataFrame(; params, Dss, retcodes, final_states, final_Ts, maxresids, num_surv)
    jldsave(outfname; metadata, df)
    df
end

################################################################################
# Fitting the MM d
################################################################################
"""
    fit_mm_d(si_params, si_fs, si_Ds, K, l, p, ks, kweight=1e5)

Fit the MM d by matching its disprel peak to that of the given SI system, both peaks
located numerically. Returns (; fit_d, si_mrls, mmls, si_k, mm_k, si_h, mm_h) with the
disprels evaluated on the passed ks and si_k/mm_k, si_h/mm_h the locations and heights
of the two peaks, or nothing if the SI system is not spatially unstable.
"""
function fit_mm_d(si_params, si_fs, si_Ds, K, l, p, ks, kweight=1e5;
    m=1., c=1., DN=0., DI=1., extinct_threshold=1e-9, logdlims=(-6., 5.),
)
    si_mrls = linstab_simple(si_params, si_Ds, si_fs, ks)
    si_h0, i = findmax(si_mrls)
    si_h0 <= 0. && return nothing

    si_opt = optimize(log(ks[max(i-1, 1)]), log(ks[min(i+1, end)])) do logk
        -linstab_simple(si_params, si_Ds, si_fs, [exp(logk)])[1]
    end
    si_k, si_h = exp(Optim.minimizer(si_opt)), -Optim.minimum(si_opt)

    mm_peak = mmp -> begin
        hss = mmv3_get_hss_unique(mmp)
        (isempty(hss) || hss[1] < extinct_threshold) && return nothing
        j = findmax(fr3_disprel_simple(mmp, DN, DI, p, hss[1], ks))[2]
        mm_opt = optimize(log(ks[max(j-1, 1)]), log(ks[min(j+1, end)])) do logk
            -fr3_disprel_simple(mmp, DN, DI, p, hss[1], [exp(logk)])[1]
        end
        exp(Optim.minimizer(mm_opt)), -Optim.minimum(mm_opt)
    end

    obj = logd -> begin
        pk = mm_peak(MMParams(; K, l, m, c, d=10^logd))
        isnothing(pk) && return Inf
        mm_k, mm_h = pk
        abs(mm_h - si_h) / abs(si_h) + kweight * abs(mm_k - si_k) / abs(si_k)
    end

    opt_r = optimize(obj, logdlims...)
    if opt_r.minimizer in logdlims
        @warn "Hitting logdlims in optimization"
    end
    d = 10^Optim.minimizer(opt_r)

    mmp = MMParams(; K, l, m, c, d)
    mmls = fr3_disprel_simple(mmp, DN, DI, p, mmv3_get_hss_unique(mmp)[1], ks)
    mm_k, mm_h = mm_peak(mmp)
    (; fit_d=d, si_mrls, mmls, si_k, mm_k, si_h, mm_h)
end

function process_outdir1(dirpath, ks, kweight=0., DN=0.;
    out_fname=nothing,
    thr_maxresid=1e-5,
    thr_hss_mrl=1e-9,
)
    processing_params = (;
        dirpath, ks, kweight, DN, thr_maxresid, thr_hss_mrl
    )

    # Find all relevant files
    files = filter(readdir(dirpath; join=true)) do f
        endswith(f, ".jld2") && !endswith(f, "_fit.jld2")
    end
    sort!(files; by=f -> parse(Int, match(r"gi(\d+)", basename(f))[1]))
    @printf "Reading %d files\n" length(files)

    # Combine data from all files into one df
    adf = mapreduce(vcat, files) do fname
        metadata, df = load(fname, "metadata", "df")
        insertcols!(df, 1,
            :file => basename(fname),
            :row_id => 1:nrow(df),
            :K => metadata.K,
            :l => metadata.l,
            :p => metadata.p,
        )
    end
    @printf "Files have nrow(adf) = %d runs\n" nrow(adf)

    # Calculate hss stability for later use
    adf.hss_mrl = map(eachrow(adf)) do r
        M1 = Matrix{Float64}(undef, sum(get_Ns(r.params)), sum(get_Ns(r.params)))
        make_M1!(M1, r.params, r.final_states)
        maximum(real, eigvals!(M1))
    end

    # Filter for good data only
    keep_row = map(eachrow(adf)) do r
        (r.retcodes == ReturnCode.Success) &&
            (r.maxresids < thr_maxresid) &&
            (r.hss_mrl < thr_hss_mrl) &&
            (r.num_surv > 0)
    end
    adf = adf[keep_row, :];
    @printf "Keeping %d of them after filtering for successful solves, small maxresids, hss stability and non-extinctions\n" count(keep_row)

    # Set the strain diffusion rates
    for r in eachrow(adf)
        N = get_Ns(r.params)[1]
        r.Dss[1:N] .= DN
    end

    # Do the MM fits
    fits = Vector{Any}(undef, nrow(adf))
    prog = Progress(nrow(adf))
    @localize adf @tasks for i in 1:nrow(adf)
        r = adf[i, :]
        fits[i] = if (r.retcodes == ReturnCode.Success) && (r.num_surv != 0)
            fit_mm_d(r.params, r.final_states, r.Dss, r.K, r.l, r.p, ks, kweight; DN)
        end
        next!(prog)
    end
    finish!(prog)

    for c in (:fit_d, :si_k, :mm_k, :si_h, :mm_h, :si_mrls, :mmls)
        adf[!, c] = [isnothing(f) ? missing : getproperty(f, c) for f in fits] # fit_mm_d returns nothing when the SI system is not spatially unstable
    end
    @printf "Of the %d filtered runs, %d were spatiall unstable and have a MM fit\n" nrow(adf) count(!ismissing, adf.fit_d)

    if !isnothing(out_fname)
        jldsave(out_fname; adf, processing_params)
    end

    adf, processing_params
end

################################################################################
# Runs
################################################################################
function main1()
    Klps_to_run = [(K, l, p) for p in [0.01, 0.1, 1.] for l in [0.75, 0.9, 0.99, 0.999] for K in range(10^0.5, 10^2, 5)]
    mkpath("main1")
    for (gi, (K, l, p)) in enumerate(Klps_to_run)
        @printf("Running %d/%d: K=%.3f, l=%.3f, p=%.3f\n", gi, length(Klps_to_run), K, l, p)
        flush(stdout)

        solve_si_odes("main1/gi$(gi).jld2", 25,
            K, l, p,
            1e8, 1e-9,
        )
    end
end

function main2()
    Klps_to_run = [(K, l, p) for p in [0.01, 0.1, 1.] for l in [0.75, 0.9, 0.99, 0.999] for K in range(10^0.5, 10^2, 5)]
    mkpath("main2")
    for (gi, (K, l, p)) in enumerate(Klps_to_run)
        @printf("Running %d/%d: K=%.3f, l=%.3f, p=%.3f\n", gi, length(Klps_to_run), K, l, p)
        flush(stdout)

        solve_si_odes("main2/gi$(gi).jld2", 100,
            K, l, p,
            1e8, 1e-9,
        )
    end
end

function main3()
    Klps_to_run = [(K, l, p) for p in [0.01, 0.1, 1.] for l in [0.999] for K in [10.]]
    mkpath("main3_one_Kl_point")
    for (gi, (K, l, p)) in enumerate(Klps_to_run)
        @printf("Running %d/%d: K=%.3f, l=%.3f, p=%.3f\n", gi, length(Klps_to_run), K, l, p)
        flush(stdout)

        solve_si_odes("main3_one_Kl_point/gi$(gi).jld2", 100,
            K, l, p,
            1e8, 1e-9,
        )
    end
end

function main4()
    Klps_to_run = [(K, l, p) for p in [0.01, 0.1, 1.] for l in [0.999] for K in [10.]]
    mkpath("main4_one_Kl_point")
    for (gi, (K, l, p)) in enumerate(Klps_to_run)
        @printf("Running %d/%d: K=%.3f, l=%.3f, p=%.3f\n", gi, length(Klps_to_run), K, l, p)
        flush(stdout)

        solve_si_odes("main4_one_Kl_point/gi$(gi).jld2", 1000,
            K, l, p,
            1e8, 1e-9,
        )
    end
end

function main5_pd_cov1()
    Ks = 10 .^ range(0., 4.0, 20)
    leak_xs = range(0.0, LeakageScale.ltox(0.999), 10)
    lis = LeakageScale.l.(leak_xs)

    Klps_to_run = [(K, l, p) for p in [0.01, 0.1, 1.] for l in lis for K in Ks]

    mkpath("main5_pd_cov1")
    for (gi, (K, l, p)) in enumerate(Klps_to_run)
        @printf("Running %d/%d: K=%.3f, l=%.3f, p=%.3f\n", gi, length(Klps_to_run), K, l, p)
        flush(stdout)

        solve_si_odes("main5_pd_cov1/gi$(gi).jld2", 100,
            K, l, p,
            1e8, 1e-9,
        )
    end
end
