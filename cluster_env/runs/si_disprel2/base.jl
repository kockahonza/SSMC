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
