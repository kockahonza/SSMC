using Revise
using SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.MinimalModelV2

include("../../../scripts/single_influx.jl")

using Base.Threads, OhMyThreads
using Printf
using ProgressMeter
using JLD2, Geppetto
using Random, Distributions
using DataFrames, DataFramesMeta
using Optim

################################################################################
# Stage 1: generate SI systems and find their well-mixed steady states
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
    surv_threshold=1e-9,
)
    rsg = get_si_sampler_for_paper(K, l, DN; DR=p, N, M)
    N = rsg.Ns
    metadata = (;
        num_runs, K, l, p, T, abstol, reltol, DN, N, M, si_u0, solver, maxtime, surv_threshold,
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
            callback=CallbackSet(make_timer_callback(maxtime), make_ode_extinction_exit_callback(N, abstol / 10), PositiveDomain(si_u0)),
            abstol=abstol,
            reltol=reltol,
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
# Stage 2: dispersion relations and fitting the MM d
################################################################################
function add_disprels!(df, ks; ls_threshold=1e-9)
    df.ls_evals = tmap(eachrow(df)) do r
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

# peak location and height, refined by fitting a parabola around the grid maximum
function peak_metrics(ks, mrls)
    h, i = findmax(mrls)
    if 1 < i < length(mrls)
        x1, x2, x3 = ks[i-1], ks[i], ks[i+1]
        y1, y2, y3 = mrls[i-1], mrls[i], mrls[i+1]
        denom = (x2 - x1) * (y2 - y3) - (x2 - x3) * (y2 - y1)
        if denom != 0.
            k = x2 - 0.5 * ((x2 - x1)^2 * (y2 - y3) - (x2 - x3)^2 * (y2 - y1)) / denom
            a = ((y1 - y2) / (x1 - x2) - (y2 - y3) / (x2 - x3)) / (x1 - x3)
            return k, y2 - a * (x2 - k)^2
        end
    end
    ks[i], h
end

function make_mm_disprel(K, l, p, d, ks; DN=0., test_threshold=1e-9)
    mmp = MMParams(; K, l, m=1., c=1., d)
    mmicrm_params = mmp_to_mmicrm(mmp; static=false)
    mm_fs = get_mm_the_nonext_sol(mmp)
    mresid = mmicrmmaxresid(mm_fs, mmicrm_params)
    if (mresid > test_threshold) || (mm_fs[1] < test_threshold)
        return nothing
    end
    linstab_simple(mmicrm_params, [DN, 1., p], mm_fs, ks)
end

function fit_ds!(df, ks, K, l, p;
    DN=0.,
    test_threshold=1e-9,
    kweight=1.,
)
    opt_rs = tmap(eachrow(df)) do r
        si_k, si_h = peak_metrics(ks, r.mrls)
        optimize([1.]) do u
            mm_mrls = make_mm_disprel(K, l, p, u[1], ks; DN, test_threshold)
            isnothing(mm_mrls) && return Inf
            mm_k, mm_h = peak_metrics(ks, mm_mrls)
            obj = abs(mm_h - si_h) / abs(si_h)
            if si_k > 0.
                obj += kweight * abs(mm_k - si_k) / si_k
            end
            obj
        end
    end

    df.fit_opt_rs = opt_rs
    df.fit_ds = map(opt_r -> opt_r.minimizer[1], opt_rs)
    df.fit_quality = map(opt_r -> opt_r.minimum, opt_rs)
    df.mm_mrls = tmap(df.fit_ds) do d
        make_mm_disprel(K, l, p, d, ks; DN, test_threshold)
    end
    df
end

get_naive_mm_mrls(ks, K, l, p; DN=0.) = make_mm_disprel(K, l, p, 1., ks; DN)

# fit a saved stage 1 file, saving to <name><suffix>.jld2 (and returning the df)
function fit_file(fname, ks; suffix="_fit", kwargs...)
    metadata, df = load(fname, "metadata", "df")
    add_disprels!(df, ks)
    fit_ds!(df, ks, metadata.K, metadata.l, metadata.p; DN=metadata.DN, kwargs...)
    jldsave(replace(fname, ".jld2" => "$(suffix).jld2"); metadata, ks, df)
    df
end

function fit_dir(dir, ks; suffix="_fit", kwargs...)
    fnames = filter(readdir(dir; join=true)) do f
        endswith(f, ".jld2") && !endswith(f, "$(suffix).jld2")
    end
    for (i, fname) in enumerate(fnames)
        @printf("Fitting %d/%d: %s\n", i, length(fnames), fname)
        flush(stdout)
        fit_file(fname, ks; suffix, kwargs...)
    end
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

        solve_si_odes("main1/gi$(gi).jld2", 20,
            K, l, p,
            1e8, 1e-9,
        )
    end
end

fit_main1(ks=range(0.01, 100, 1000); kwargs...) = fit_dir("main1", ks; kwargs...)

################################################################################
# Plotting
################################################################################
function plot_fitted_disprels(df, ks, num_runs=1;
    autoylims=true,
    legend=false,
    same_col=true,
    figure=(;),
    axis=(;),
)
    fig = Figure(; figure...)
    ax = Axis(fig[1,1]; axis...)

    iis = sample(1:nrow(df), num_runs; replace=false)

    for (ci, ri) in enumerate(iis)
        r = df[ri,:]
        lines!(ax, ks, r.mrls;
            color=Cycled(ci),
            label=(@sprintf "ri=%d, fitted d=%.3g" ri r.fit_ds)
        )
        lines!(ax, ks, r.mm_mrls;
            color=same_col ? Cycled(ci) : :black,
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

    FigureAxisAnything(fig, ax, nothing)
end
