using Revise
using SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.MinimalModelV2, SSMCMain.ModifiedMiCRM.TwoMMs

include("../../../scripts/single_influx.jl")

using Base.Threads, OhMyThreads
using Printf
using ProgressMeter
using JLD2, Geppetto
using Random, Distributions
using DataFrames, DataFramesMeta
using Optim

################################################################################
# Finding SI steady states
################################################################################
function solve_si_odes(
    outfname, num_runs,
    K, l, p,
    T, tol;
    DN=0.,
    N=20, M=20,
    si_u0=[fill(1., N); fill(0., M)],
    solver=TRBDF2,
    maxtime=30.,
    surv_threshold=1e-9,
)
    rsg = get_si_sampler_for_paper(K, l, DN; DR=p, N, M)
    N = rsg.Ns
    metadata = (;
        num_runs, K, l, p, T, tol, DN, N, M, si_u0, solver, maxtime, surv_threshold,
        rsg,
    )

    params = Vector{BMMiCRMParams}(undef, num_runs)
    Dss = Vector{Vector{Float64}}(undef, num_runs)
    retcodes = Vector{ReturnCode.T}(undef, num_runs)
    final_states = Vector{Vector{Float64}}(undef, num_runs)
    maxresids = Vector{Float64}(undef, num_runs)
    num_surv = Vector{Int}(undef, num_runs)

    prog = Progress(num_runs)
    @tasks for i in 1:num_runs
        gen_ps = rsg()
        si_ps = gen_ps.mmicrm_params
        si_Ds = gen_ps.Ds

        si_p = make_mmicrm_problem(si_ps, si_u0, T)
        si_s = solve(si_p, solver();
            dense=false,
            save_everystep=false,
            callback=CallbackSet(make_timer_callback(maxtime), make_ode_extinction_exit_callback(N, tol / 10), PositiveDomain(si_u0)),
            abstol=tol,
            reltol=tol,
        )
        si_fs = si_s.u[end]

        params[i] = si_ps
        Dss[i] = si_Ds
        retcodes[i] = si_s.retcode
        final_states[i] = si_fs
        maxresids[i] = mmicrmmaxresid(si_s)
        num_surv[i] = count(>(surv_threshold), si_fs[1:N])

        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)

    df = DataFrame(; params, Dss, retcodes, final_states, maxresids, num_surv)
    jldsave(outfname; metadata, df)
    df
end

################################################################################
# Getting dispersion relations for SI
################################################################################
function add_disprels!(df, ks; ls_threshold=1e-9)
    df.ls_evals = map(eachrow(df)) do r
        linstab_simple(r.params, r.Dss, r.final_states, ks; returnobj=:evals)
    end
    df.ls_revals = real(df.ls_evals)
    df.mrls = map(df.ls_revals) do revals
        getindex.(revals, 1)
    end
    df.mmrls = maximum.(df.mrls)
    df.spatially_unstable = df.mmrls .> ls_threshold
    df
end

################################################################################
# Minimal model stuff
################################################################################
function solve_mm(T, K, l, p, d=1.;
    m=1., c=1.,
    DN=0.,
    mm_u0=[1., 0., 0.],
    tol=1e-9,
    maxtime=10.,
)
    mmp = MMParams(;
        K, l,
        m=1., c=1.,
        d=d,
    )
    mm_ps = mmp_to_mmicrm(mmp; static=false)

    mm_p = make_mmicrm_problem(mm_ps, mm_u0, T)
    solve(mm_p, QNDF();
        callback=CallbackSet(make_timer_callback(maxtime), PositiveDomain(mm_u0)),
        abstol=tol,
        reltol=tol,
    )
end

function fit_ds!(df, ks, T, K, l, p;
    DN=0.,
    test_threshold=1e-9,
    kwargs...
)
    add_disprels!(df, ks)

    mm_Ds = [DN, 1., p]

    opt_rs = map(eachrow(df)) do r
        optimize([1.]) do u
            mm_s = solve_mm(T, K, l, p, u[1]; DN, kwargs...)

            mm_fs = mm_s.u[end]
            mresid = mmicrmmaxresid(mm_s)
            if (mresid > test_threshold) || (mm_fs[1] < test_threshold)
                return Inf
            end

            mm_mrls = linstab_simple(mm_s.prob.p, mm_Ds, mm_fs, ks)
            abs(maximum(mm_mrls) - maximum(r.mrls))
        end
    end
    df.fit_opt_rs = opt_rs

    df.fit_ds = map(opt_rs) do opt_r opt_r.minimizer[1] end

    df
end

function add_mm_mrls!(df, ks, T, K, l, p;
    DN=0.,
    kwargs...
)
    mm_Ds = [DN, 1., p]

    df.mm_mrls = map(eachrow(df)) do r
        mm_s = solve_mm(T, K, l, p, r.fit_ds; DN, kwargs...)
        linstab_simple(mm_s.prob.p, mm_Ds, mm_s.u[end], ks)
    end

    df
end

################################################################################
# Cluster runs
################################################################################
function main1()
    Klps_to_run = []
    for p in [1., 0.1]
        for l in [1., 0.999, 0.99, 0.9, 0.8]
            for K in range(10^0.5, 10^2, 10)
                push!(Klps_to_run, (K, l, p))
            end
        end
    end
    Klps_to_run
    for ri in 1:length(Klps_to_run)
        K, l, p = Klps_to_run[ri]
        @printf("Running %d/%d: K=%.3f, l=%.3f, p=%.3f\n", ri, length(Klps_to_run), K, l, p)
        flush(stdout)

        solve_si_odes("main1_$(ri).jld2", 100,
            K, l, p,
            1e8, 1e-9,
        )
    end
end

function main2()
    Klps_to_run = []
    for p in [1.]
        for l in [0.999]
            for K in range(10^0.5, 10^2, 5)
                push!(Klps_to_run, (K, l, p))
            end
        end
    end
    Klps_to_run
    for ri in 1:length(Klps_to_run)
        K, l, p = Klps_to_run[ri]
        @printf("Running %d/%d: K=%.3f, l=%.3f, p=%.3f\n", ri, length(Klps_to_run), K, l, p)
        flush(stdout)

        solve_si_odes("main2/ri$(ri).jld2", 100,
            K, l, p,
            1e8, 1e-9,
        )
    end
end

function main3()
    Klps_to_run = []
    for p in [1.]
        for l in [0.999]
            for K in range(10^0.5, 10^2, 5)
                push!(Klps_to_run, (K, l, p))
            end
        end
    end
    Klps_to_run
    for gi in 1:length(Klps_to_run)
        K, l, p = Klps_to_run[gi]
        @printf("Running %d/%d: K=%.3f, l=%.3f, p=%.3f\n", gi, length(Klps_to_run), K, l, p)
        flush(stdout)

        solve_si_odes("main3/gi$(gi).jld2", 1000,
            K, l, p,
            1e8, 1e-9,
        )
    end
end

################################################################################
# Plotting
################################################################################
function plot_disprels(df, ks, num_runs=nrow(df);
    num_evals=1,
    autoylims=true,
    random_rows=true,
)
    fig = Figure()
    ax = Axis(fig[1,1])

    iis = if random_rows
        sample(1:nrow(df), num_runs; replace=false)
    else
        1:num_runs
    end

    for ri in iis
        r = df[ri,:]
        lines!(ax, ks, r.mrls; color=Cycled(ri))
        if num_evals > 1
            for ii in 2:num_evals
                lines!(ax, ks, getindex.(r.ls_revals, ii); color=Cycled(ri), linestyle=:dash)
            end
        end
    end

    if autoylims
        mm = abs(maximum(df[iis,:].mmrls))
        ylims!(ax, -1.1mm, 1.1mm)
    end

    fig
end

function plot_fitted_disprels(df, ks, num_runs=1;
    autoylims=true,
    legend=false,
)
    fig = Figure()
    ax = Axis(fig[1,1])

    iis = sample(1:nrow(df), num_runs; replace=false)

    for (ci, ri) in enumerate(iis)
        r = df[ri,:]
        lines!(ax, ks, r.mrls;
            color=Cycled(ci),
               label=(@sprintf "ri=%d, fited d=%.3g" ri r.fit_ds)
        )
        lines!(ax, ks, r.mm_mrls;
            color=:black,
            linestyle=:dash
        )
    end

    if legend
        axislegend(ax)
    end

    if autoylims
        mm = abs(maximum(df[iis,:].mmrls))
        ylims!(ax, -1.1mm, 1.1mm)
    end

    fig
end
