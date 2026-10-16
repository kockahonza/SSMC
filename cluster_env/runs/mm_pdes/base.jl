using Revise
using SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.MinimalModelV3, SSMCMain.ModifiedMiCRM.FFTAnalysis

using Base.Threads, OhMyThreads
using Printf
using ProgressMeter
using JLD2
using Random
using DataFrames, DataFramesMeta
using LinearAlgebra

BLAS.set_num_threads(1)

################################################################################
# Stage 1: the setup - homogeneous steady state, linstab, L and the u0s
################################################################################
"""
Does a simple sampled linear stability analysis (using the analytic fast
resource formula), finds the peak and returns it along with some checks.
"""
function mm_peak_k(mmp::MMParams, DN, DI, p, hss_N, ks)
    mrls = fr3_disprel_simple(mmp, DN, DI, p, hss_N, ks)
    h, i = findmax(mrls)
    (;
        peak_k=ks[i], peak_h=h,
        at_edge=(i == 1) || (i == length(ks)),
        unstable=h > 0.0,
        mrls,
    )
end

"""
Domain size holding exactly `num_initial_peaks` wavelengths of wavenumber `k`.
"""
choose_L(k, num_initial_peaks) = 2pi * num_initial_peaks / k

"""
Prepares a set of minimal model PDE runs for a range of ps with multiple
randomly perturbed initial conditions for each p. Well-mixed steady states
and linear stability (to set domain size) are analytic. Produces a
dataframe with all that is needed to run many PDE computations.
"""
function make_setup(outfname;
    K=15.0, l=0.999, m=1.0, c=1.0, d=1.0,
    DN=1e-6, DI=1.0,
    ps=10 .^ range(-2, 0, 5),
    num_reps=10,
    sN=2500,
    min_peaks=10,
    epsilon=1e-3,
    T=1e8,
    ks=10 .^ range(-6, 4, 20001),
    seed=20260820,
)
    mmp = MMParams(; K, l, m, c, d)
    hss = mmv3_get_hss_unique(mmp)
    mmicrm_params = mmp_to_mmicrm(mmp; static=false)

    @printf("MM: K=%.4g l=%.4g m=%.4g c=%.4g d=%.4g\n", K, l, m, c, d)
    @printf("well-mixed steady state: N=%.6g I=%.6g R=%.6g\n", hss...)
    @printf("DN=%.4g DI=%.4g, sN=%d, %d peaks minimum, %d reps per p\n\n",
        DN, DI, sN, min_peaks, num_reps)

    # Linear stability of every p, and the L that follows from it
    lss = [mm_peak_k(mmp, DN, DI, p, hss[1], ks) for p in ps]
    Ls = [choose_L(ls.peak_k, min_peaks) for ls in lss]

    @printf("%12s %12s %12s %12s %12s %12s  %s\n",
        "p", "peak_k", "peak_h", "wavelength", "L", "dx", "linstab")
    for (p, ls, L) in zip(ps, lss, Ls)
        @printf("%12.6g %12.6g %12.6g %12.6g %12.6g %12.6g  %s\n",
            p, ls.peak_k, ls.peak_h, 2pi / ls.peak_k, L, L / sN,
            ls.unstable ? (ls.at_edge ? "UNSTABLE (peak at ks edge!)" : "unstable") : "STABLE")
    end

    stable = findall(ls -> !ls.unstable, lss)
    at_edge = findall(ls -> ls.unstable && ls.at_edge, lss)
    if !isempty(stable)
        error("spatially STABLE at p = $(ps[stable]); nothing to run there")
    end
    if !isempty(at_edge)
        error("disprel peak sits on the edge of ks at p = $(ps[at_edge]); widen ks")
    end
    @printf("\nall %d p values are spatially unstable, %d runs to do\n",
        length(ps), length(ps) * num_reps)

    # The u0s. `perturb_u0_uniform` draws from the global rng, so seeding it is
    # what makes the file reproducible; drawing in row order means adding reps
    # at the end does not disturb the earlier ones.
    Random.seed!(seed)
    hss_field = expand_u0_to_size((sN,), hss)

    rows = NamedTuple[]
    for (pi_, (p, ls, L)) in enumerate(zip(ps, lss, Ls))
        for rep in 1:num_reps
            u0 = perturb_u0_uniform(1, 2, copy(hss_field), epsilon)
            clamp!(u0, 0.0, Inf)
            push!(rows, (;
                p_i=pi_, p, rep,
                mmp, mmicrm_params, Ds=[DN, DI, p],
                hss, peak_k=ls.peak_k, peak_h=ls.peak_h,
                L, sN, T, u0,
            ))
        end
    end
    pde_df = DataFrame(rows)

    metadata = (;
        K, l, m, c, d, DN, DI, ps, num_reps, sN, min_peaks, epsilon, T, ks, seed,
        mmp, hss, Ls,
        mrls=[ls.mrls for ls in lss],
        peak_ks=[ls.peak_k for ls in lss],
        peak_hs=[ls.peak_h for ls in lss],
    )
    jldsave(outfname; metadata, pde_df)
    @printf("wrote %s (%d rows)\n", outfname, nrow(pde_df))
    flush(stdout)

    pde_df
end

################################################################################
# Stage 2: PDE solving
################################################################################
function run_mm_pde(mmicrm_params, Ds, u0, T, L, save_step;
    abstol=1e-7,
    reltol=1e-9,
    solver=QNDF,
    maxtime=60 * 60,
    solver_threads=nothing,
    positive_domain=true,
    kwargs...
)
    sN = size(u0, 2)
    dx = L / sN

    sps = BSMMiCRMParams(mmicrm_params, Ds, CartesianSpace{1,Tuple{Periodic}}(SA[dx]), solver_threads)
    sp = make_smmicrm_problem(sps, copy(u0), T; jac_type=:sparse)

    saved_ts, saved_us, scb = make_stepped_saver_callback(u0, save_step)
    push!(saved_ts, 0.0)
    push!(saved_us, copy(u0))

    s = solve(sp, solver();
        dense=false,
        save_everystep=false,
        calck=false,
        abstol,
        reltol,
        callback=CallbackSet(make_timer_callback(maxtime),
            (positive_domain ? (PositiveDomain(copy(u0); save=false),) : ())..., scb),
        kwargs...
    )

    push!(saved_ts, s.t[end])
    push!(saved_us, s.u[end])

    (; s, saved_ts, saved_us)
end

"""
Runs the PDEs specified in a setup file as produced by `make_setup`, saving all
outputs into `outdir`. Each row of the setup dataframe is a separate PDE run,
saved in a separate file. Also produces a `settings.jld2` file with the parameters
used for the run.
"""
function run_pde_setup(setup_fname, outdir, save_step;
    abstol=1e-7,
    reltol=1e-9,
    solver=QNDF,
    maxtime=60 * 60,
    rows=nothing,
    overwrite=false,
    run_threads=nthreads(),
    solver_threads=nothing,
    kwargs...
)
    setup = load(setup_fname)
    pde_df = setup["pde_df"]
    rows = isnothing(rows) ? (1:nrow(pde_df)) : rows
    mkpath(outdir)

    settings = (;
        setup_fname, abstol, reltol, solver, maxtime, save_step,
        run_threads, solver_threads,
        setup_metadata=setup["metadata"],
    )
    jldsave(joinpath(outdir, "settings.jld2"); settings)

    todo = [i for i in rows if overwrite || !isfile(joinpath(outdir, @sprintf("row%04d.jld2", i)))]
    @printf("%d/%d rows to run (%d already done), %d at a time with %s thread(s) each\n",
        length(todo), length(rows), length(rows) - length(todo),
        run_threads, isnothing(solver_threads) ? "1" : string(solver_threads))
    flush(stdout)

    prog = Progress(length(todo))
    @tasks for i in todo
        @set ntasks = run_threads

        r = pde_df[i, :]
        try
            start_time = time()
            (; s, saved_ts, saved_us) = run_mm_pde(r.mmicrm_params, r.Ds, r.u0, r.T, r.L, save_step;
                abstol, reltol, solver, maxtime, solver_threads,
                kwargs...
            )
            realtime = time() - start_time

            jldsave(joinpath(outdir, @sprintf("row%04d.jld2", i));
                pde_df_row=i, p_i=r.p_i, p=r.p, rep=r.rep,
                L=r.L, sN=r.sN, T=r.T, peak_k=r.peak_k, peak_h=r.peak_h,
                retcode=s.retcode,
                final_state=s.u[end],
                final_T=s.t[end],
                maxresid=smmicrmmaxresid(s),
                num_saved=length(saved_ts),
                realtime,
                saved_ts,
                saved_us,
                settings,
            )
            s = nothing
        catch e
            @printf("Row %d failed: %s\n", i, sprint(showerror, e))
        end
        GC.gc()

        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)

    outdir
end

"""
Load everything in the row files into a df sorted by `pde_df_row`. `states`
keeps the saved time series, `setup` joins on the setup file's columns.
"""
function collect_pde_runs(outdir; states=true, setup=true)
    fnames = sort(filter(f -> startswith(f, "row") && endswith(f, ".jld2"), readdir(outdir)))
    df = DataFrame(map(fnames) do f
        jldopen(joinpath(outdir, f)) do d
            (; (Symbol(k) => d[k] for k in keys(d) if states || !(k in ("saved_ts", "saved_us")))...)
        end
    end)
    isempty(df) && return df

    if setup
        # adds the setup columns not in the row files (mmicrm_params, Ds, u0, ...)
        pde_df = load(df.settings[1].setup_fname, "pde_df")
        pde_df.pde_df_row = 1:nrow(pde_df)
        df = leftjoin(df, select(pde_df, :pde_df_row, setdiff(names(pde_df), names(df))); on=:pde_df_row)
    end
    sort!(df, :pde_df_row)
end

"""
Loads the u and t time series from a single PDE run directly.
"""
function load_pde_states(outdir, i)
    jldopen(joinpath(outdir, @sprintf("row%04d.jld2", i))) do d
        (; ts=d["saved_ts"], us=d["saved_us"])
    end
end

################################################################################
# Analysis v2: FFT analysis and finding peaks
################################################################################
function add_extras_and_save(outdir, outfname)
    df = collect_pde_runs(outdir);

    df.early_time_modes = map(eachrow(df)) do r
        first_strong_mode(map(r.saved_us) do u
                @view u[1, :]
            end, r.L)
    end;
    @show count(isnothing, df.early_time_modes)

    df.peaks = @showprogress map(eachrow(df)) do r
        map(r.saved_us) do u
            find_peaks_periodic(@view u[1, :];
                min_prom=0.1,
            )
        end
    end;

    jldsave(outfname; df)
end
