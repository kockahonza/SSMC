using Revise
using SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.MinimalModelV3

using Base.Threads, OhMyThreads
using Printf
using ProgressMeter
using JLD2
using Random
using DataFrames, DataFramesMeta

################################################################################
# Stage 1: the setup - homogeneous steady state, linstab, L and the u0s
################################################################################
"""
    mm_peak_k(mmp, DN, DI, p, hss_N, ks)

Where the MM dispersion relation peaks at this `p`, found by scanning `ks` and
taking the largest value - no root finding, so the k is only good to the grid
spacing. That is all `choose_L` needs, since it rounds to a whole number of
peaks anyway.

Returns `(; peak_k, peak_h, at_edge, unstable)`. `unstable` is whether the
disprel gets above zero at all; when it does not the k is meaningless and only
`unstable` should be read. `at_edge` marks a peak sitting on the first or last
k, where the real maximum is outside the scanned range - widen `ks` if it shows
up.
"""
function mm_peak_k(mmp::MMParams, DN, DI, p, hss_N, ks)
    mrl = fr3_disprel_simple(mmp, DN, DI, p, hss_N, ks)
    h, i = findmax(mrl)
    (;
        peak_k=ks[i], peak_h=h,
        at_edge=(i == 1) || (i == length(ks)),
        unstable=h > 0.0,
        mrl,
    )
end

"""
    num_peaks_in(L, k)

How many peaks a mode of wavenumber `k` puts in a periodic domain of length `L`.
The one place the peaks <-> wavenumber convention lives, so `choose_L` and any
later check of what it returned cannot disagree about a factor of 2pi.
"""
num_peaks_in(L, k) = L * k / (2pi)

"""
    choose_L(k, min_peaks)

Domain size holding exactly `min_peaks` peaks of wavenumber `k`, ie the
inverse of [`num_peaks_in`](@ref).
"""
choose_L(k, min_peaks) = 2pi * min_peaks / k

"""
    make_setup(outfname; kwargs...)

Build the run plan and write it to `outfname`. Everything here is analytic and
takes well under a second, but it is still a separate stage with its own file:
the u0s are random, so pinning them down once and shipping the file to the
cluster is what makes the runs reproducible and lets a resubmit continue the
same runs rather than new ones.

For each `p` in `ps`:
- the well-mixed steady state from `mmv3_get_hss_unique`, ie the analytic
  quadratic root, not an ODE solve. It does not depend on `p`, so it is
  computed once and shared.
- the dispersion relation peak, which does, and from it `L = 2pi * min_peaks /
  peak_k` so that at least `min_peaks` peaks fit in the domain.
- `num_reps` initial states, each the steady state expanded over `sN`
  gridpoints and perturbed by `perturb_u0_uniform` with an absolute `epsilon`,
  clamped at 0.

Returns the `pde_df` it wrote, one row per (p, rep). Prints a table of the
linear stability of every p and **throws** if any of them is spatially stable -
a run started from a stable state would just relax back to homogeneous and
tell us nothing, so it is a mistake to spend cluster time on it. Same for a
peak found at the edge of `ks`, where the L would be wrong.
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
        mrls=[ls.mrl for ls in lss],
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
"""
    make_progress_callback(label; every=60.0)

Print where a solve has got to every `every` seconds of wall clock: the time
reached, the current step size, the step count and the elapsed time.

Exists because a long solve is otherwise completely opaque - the row file is
only written at the end, so a run that will take a week looks exactly like one
that will take an hour. `t` here is the thing to watch: these solves spend
almost all of their steps in the early, dynamic decades and then cross the
remaining ones in a handful of steps, so progress is very far from linear in
`t`. Read it as "which decade are we in", and compare `dt` against the time
left - once `dt` is within a few decades of `T - t` the end is near.

Holds mutable state, so make a fresh one per run.
"""
function make_progress_callback(label; every=60.0)
    t0 = time()
    nextat, nsteps = Ref(t0 + every), Ref(0)
    DiscreteCallback(
        function (u, t, _)
            nsteps[] += 1
            time() >= nextat[]
        end,
        function (i)
            @printf("[%s] t=%.3e  dt=%.3e  steps=%d  elapsed=%.0fs\n",
                label, i.t, i.dt, nsteps[], time() - t0)
            flush(stdout)
            nextat[] = time() + every
        end;
        save_positions=(false, false),
    )
end

"""
    run_mm_pde(mmicrm_params, Ds, u0, T, L; kwargs...)

Solve one 1d MM PDE on a periodic domain of length `L` with `size(u0, 2)`
gridpoints, from `u0` to time `T`.

Takes `mmicrm_params` and `Ds` separately since that is how the setup file
stores them - the `BSMMiCRMParams` gluing them to a space is built here, where
`dx = L / sN` is known.

Solver setup is si_pdes1's: `QNDF` with a loose `abstol` and a tight `reltol`,
plus `PositiveDomain` on top, which is the combination that got these systems
to a steady state in sane wall clock without the solver wandering negative.
`save=false` on `PositiveDomain` keeps it from saving every step it touches.

`positive_domain=false` drops it entirely. Worth trying on long-T runs: once the
pattern has settled, N sits at ~1e-50 in the gaps, so roundoff there is
perpetually negative, `PositiveDomain` rejects the step and cuts dt, and dt
never grows past ~1e10 - which makes the cost linear in T rather than
logarithmic. Those values are decades below `abstol`, so nothing physical is
being protected. Check the run still ends positive before trusting it.

Returns `(; s, saved_ts, saved_us)` - the solution plus a series of states over
time for watching the coarsening, every `save_step`-th solver step. Saving
actual steps costs nothing and the states are exactly what the solver produced.
The price is an irregular time grid that differs between runs and a state count
that rides on how many steps the solver takes. At 3 x sN a state that is only
60KB here, so a run of even 100k steps at `save_step=20` holds 300MB.
"""
function run_mm_pde(mmicrm_params, Ds, u0, T, L;
    abstol=1e-7,
    reltol=1e-9,
    solver=QNDF,
    maxtime=60 * 60,
    solver_threads=nothing,
    save_step=20,
    progress_label=nothing,   # a string here turns on periodic progress printing
    progress_every=60.0,
    positive_domain=true,     # false drops the PositiveDomain callback, see below
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
            (positive_domain ? (PositiveDomain(copy(u0); save=false),) : ())..., scb,
            (isnothing(progress_label) ? () : (make_progress_callback(progress_label; every=progress_every),))...),
        kwargs...
    )

    push!(saved_ts, s.t[end])
    push!(saved_us, s.u[end])

    (; s, saved_ts, saved_us)
end

"""
    run_pde_setup(setup_fname, outdir; kwargs...)

Run every row of the `pde_df` in a setup file and write one small jld2 per row
into `outdir`.

Per-row files rather than one big one because these runs are hours each: a job
that hits its walltime keeps everything it finished, and rerunning it picks up
where it stopped - rows whose file already exists are skipped unless
`overwrite=true`. Use `rows` to split the work across array jobs. A row that
throws is reported and left without a file, so a rerun retries it.

`run_threads` rows are solved at once, each solver using `solver_threads`
threads. With 50 rows and 50 cores one row per thread and a serial solver is
the right split; only give a solver its own threads if there are more cores
than rows.

Read the results back with `collect_pde_runs`.
"""
function run_pde_setup(setup_fname, outdir;
    abstol=1e-7,
    reltol=1e-9,
    solver=QNDF,
    maxtime=60 * 60,
    save_step=20,
    rows=nothing,
    overwrite=false,
    run_threads=nthreads(),
    solver_threads=nothing,
    progress=false,   # true prints per-row progress, see make_progress_callback
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
            (; s, saved_ts, saved_us) = run_mm_pde(r.mmicrm_params, r.Ds, r.u0, r.T, r.L;
                abstol, reltol, solver, maxtime, solver_threads, save_step,
                progress_label=(progress ? @sprintf("row%04d p=%.3g", i, r.p) : nothing),
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
                # the coarsening series, nearly all of the file. Separate keys
                # so collect_pde_runs can read the rest without pulling it in.
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
    collect_pde_runs(outdir; states=false)

Gather the per-row files `run_pde_setup` wrote into one dataframe, sorted by
`pde_df_row`. The heavy `mmicrm_params`/`Ds`/`u0` stay in the setup file - join
on `pde_df_row` when they are needed.

The `saved_us` series is most of a row file, so it is left on disk unless you
ask for `states=true`; `load_pde_states` gets one row's worth, which is what
plotting actually wants. Reading is per-key so the rest of a row costs nothing.
"""
function collect_pde_runs(outdir; states=false)
    cols = ["pde_df_row", "p_i", "p", "rep", "L", "sN", "T", "peak_k", "peak_h",
        "retcode", "final_T", "maxresid", "num_saved", "realtime", "final_state"]
    states && append!(cols, ["saved_ts", "saved_us"])
    fnames = sort(filter(f -> startswith(f, "row") && endswith(f, ".jld2"), readdir(outdir)))
    df = DataFrame(map(fnames) do f
        jldopen(joinpath(outdir, f)) do d
            (; (Symbol(c) => d[c] for c in cols)...)
        end
    end)
    isempty(df) ? df : sort!(df, :pde_df_row)
end

"""
    load_pde_states(outdir, i)

The saved states of one row as `(; ts, us)`, `us[k]` being the full 3 x sN
state at `ts[k]`. Row `i` is the `pde_df_row`, ie the index into the setup
file's `pde_df`, not a position in `collect_pde_runs`.
"""
function load_pde_states(outdir, i)
    jldopen(joinpath(outdir, @sprintf("row%04d.jld2", i))) do d
        (; ts=d["saved_ts"], us=d["saved_us"])
    end
end

################################################################################
# Analysis: counting peaks and when they merged
################################################################################
"""
    count_peaks(N; thresh_frac=0.1, min_rel_amp=1e-2)

Number of peaks in a 1d periodic field, counted as upward crossings of a level
set `thresh_frac` of the way from the field's minimum to its maximum.

Crossing counting rather than local-maxima counting because it survives both
ends of a coarsening run. Any level strictly inside the field's range is
crossed upwards exactly once per period, so it gets the near-sinusoidal early
pattern right; and putting the level low (10% of the range) means a short peak
sitting next to tall ones still counts, which a mean- or midpoint-level test
would miss once the pattern has gone spiky. A local-maximum count instead
double counts any peak with a slight dip in its top.

Returns 0 for a field with no pattern in it - `min_rel_amp` is the relative
peak-to-peak amplitude below which the field counts as homogeneous, which the
seeded initial condition (1e-3 absolute on N ~ 12.9) is well under.
"""
function count_peaks(N; thresh_frac=0.1, min_rel_amp=1e-2)
    lo, hi = extrema(N)
    rng = hi - lo
    mean_abs = abs(sum(N) / length(N))
    (rng <= min_rel_amp * mean_abs) && return 0

    thresh = lo + thresh_frac * rng
    sN = length(N)
    count(i -> (N[i] > thresh) && (N[mod1(i - 1, sN)] <= thresh), 1:sN)
end

"""
    peak_series(outdir, i; field=1, kwargs...)

Peak count over time for one row, as `(; ts, np, us_field)`. `field` is which
of the 3 state components to count in - 1 is the strain N, which is where the
peaks are. `kwargs` go to [`count_peaks`](@ref).

`us_field` is the `field` row of every saved state stacked into a
`length(ts) x sN` matrix, ie exactly what a kymograph wants, so a caller that
needs both does not read the file twice.
"""
function peak_series(outdir, i; field=1, states=false, kwargs...)
    (; ts, us) = load_pde_states(outdir, i)
    np = [count_peaks(view(u, field, :); kwargs...) for u in us]
    states || return (; ts, np)

    us_field = Matrix{Float64}(undef, length(us), size(us[1], 2))
    for k in eachindex(us)
        @views us_field[k, :] .= us[k][field, :]
    end
    (; ts, np, us_field)
end

"""
    build_peak_cache(outdir; field=1, kwargs...)

Compute the peak-count series of every row once and write it to
`outdir/peak_cache.jld2`. `merge_table` picks the file up automatically if it
is there.

Worth having because the counts are what every later analysis actually wants,
and they are a ten-thousandth of the data they come from - 3.6GB of saved
states against under a megabyte of counts. The rows are also very uneven: a
median row here is 9MB and the largest is 880MB (14625 saved states), so
recomputing on every notebook run means re-reading that whole spread each time.

Rerun it if the underlying rows change; it is not invalidated automatically.
"""
function build_peak_cache(outdir; field=1, showprogress=true, kwargs...)
    fnames = sort(filter(f -> startswith(f, "row") && endswith(f, ".jld2"), readdir(outdir)))
    cache = Dict{Int,@NamedTuple{ts::Vector{Float64}, np::Vector{Int}}}()
    prog = Progress(length(fnames); enabled=showprogress)
    for f in fnames
        jldopen(joinpath(outdir, f)) do d
            cache[d["pde_df_row"]] = (;
                ts=d["saved_ts"],
                np=[count_peaks(view(u, field, :); kwargs...) for u in d["saved_us"]],
            )
        end
        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    jldsave(joinpath(outdir, "peak_cache.jld2"); cache, field)
    cache
end

"""The `build_peak_cache` file for `outdir`, or `nothing` if it has not been built."""
function peak_cache(outdir)
    f = joinpath(outdir, "peak_cache.jld2")
    isfile(f) ? load(f, "cache") : nothing
end

"""
    merge_summary(outdir, i; quiet_factor=10, kwargs...)

When one row stopped coarsening, as
`(; ts, np, max_np, final_np, t_first, t_last, final_T, quiet)`.

`t_first` is when the pattern first appears (the first time the peak count is
nonzero), `max_np` the most peaks ever counted and `t_last` the time of the
last change in the count, ie the last merge. `quiet` is whether the run then
sat at `final_np` peaks for at least `quiet_factor` times longer than it took
to get there - the test for "this is an arrested state, not a run that was
still coarsening when we stopped watching".

Note `final_T` is the solve's end time, not the last *saved* time: once the
state stops moving the solver takes enormous steps, so the saved series
naturally thins out long before `final_T`. `quiet` uses `final_T`, since the
solver having jumped straight there is itself evidence nothing was happening.
"""
function merge_summary(outdir, i; quiet_factor=10, cache=nothing, kwargs...)
    (; ts, np) = isnothing(cache) ? peak_series(outdir, i; kwargs...) : cache[i]
    final_T = jldopen(d -> d["final_T"], joinpath(outdir, @sprintf("row%04d.jld2", i)))

    nz = findfirst(>(0), np)
    t_first = isnothing(nz) ? NaN : ts[nz]
    changes = findall(k -> np[k] != np[k-1], 2:length(np)) .+ 1
    t_last = isempty(changes) ? t_first : ts[changes[end]]

    (;
        ts, np,
        max_np=maximum(np), final_np=np[end],
        t_first, t_last, final_T,
        quiet=(final_T - t_last) > quiet_factor * t_last,
    )
end

"""
    merge_table(outdir, df; kwargs...)

[`merge_summary`](@ref) for every row of a `collect_pde_runs` dataframe,
returned as a dataframe with the `p`/`rep`/`L` columns joined back on.
"""
function merge_table(outdir, df; cache=peak_cache(outdir), kwargs...)
    rows = map(eachrow(df)) do r
        ms = merge_summary(outdir, r.pde_df_row; cache, kwargs...)
        (; r.pde_df_row, r.p, r.rep, r.L, r.peak_k,
            init_peaks=num_peaks_in(r.L, r.peak_k),
            ms.max_np, ms.final_np, ms.t_first, ms.t_last, ms.final_T, ms.quiet)
    end
    DataFrame(rows)
end

"""
    peak_positions(N; thresh_frac=0.1)

Positions of the peaks in a 1d periodic field, as fractions of the domain
(0 to 1). A peak is a run of gridpoints above `thresh_frac` of the way from min
to max, and its position is that run's weighted centroid, so the answer is
sub-gridpoint. Same threshold as `count_peaks`, so the two agree on the count.
"""
function peak_positions(N; thresh_frac=0.1)
    sN = length(N)
    thr = minimum(N) + thresh_frac * (maximum(N) - minimum(N))
    above = N .> thr
    starts = [i for i in 1:sN if above[i] && !above[mod1(i - 1, sN)]]
    map(starts) do s
        idx = Int[]
        i = s
        while above[mod1(i, sN)]
            push!(idx, i); i += 1
        end
        w = [N[mod1(j, sN)] for j in idx]
        mod(sum(w .* (idx .- 0.5)) / sum(w), sN) / sN
    end
end

"""
    peak_gaps(pos)

Gaps between successive peaks on a ring, from `peak_positions` output and in
the same units. A single peak gives the whole domain.
"""
function peak_gaps(pos)
    n = length(pos)
    n == 0 && return Float64[]
    n == 1 && return [1.0]
    q = sort(pos)
    [mod(q[mod1(i + 1, n)] - q[i], 1.0) for i in 1:n]
end

"""
    peak_drift(mmicrm_params, Ds, u, L; thresh_frac=0.1)

Drift velocity of every peak, in domain lengths per unit time, from a single
state - no differencing in time.

For a profile that only translates, `dN/dt = -v dN/dx`, so `v` comes from the
residual field and the spatial gradient. It is evaluated on both flanks of a
peak at their steepest points and averaged; the two agree only for pure
translation, so their difference (`spread`) measures how much of the residual
is the peak changing shape rather than moving.

Validated against peak positions differenced over the saved series: agrees to
within ~30% for `|v|` above ~3e-14, and is noise below ~1e-16, where it bounds
the drift rather than measuring it.

Returns one `(; v, spread)` per peak, ordered as `peak_positions`.
"""
function peak_drift(mmicrm_params, Ds, u, L; thresh_frac=0.1)
    sN = size(u, 2)
    dx = L / sN
    sps = BSMMiCRMParams(mmicrm_params, Ds, CartesianSpace{1,Tuple{Periodic}}(SA[dx]), nothing)
    du = smmicrmresid(u, sps)

    N, dNdt = u[1, :], du[1, :]
    grad = [(N[mod1(i + 1, sN)] - N[mod1(i - 1, sN)]) / (2dx) for i in 1:sN]
    thr = minimum(N) + thresh_frac * (maximum(N) - minimum(N))

    starts = [i for i in 1:sN if N[i] > thr && N[mod1(i - 1, sN)] <= thr]
    map(starts) do s
        idx = Int[]
        i = s
        while N[mod1(i, sN)] > thr
            push!(idx, i); i += 1
        end
        lf = idx[argmax([grad[mod1(j, sN)] for j in idx])]
        rf = idx[argmin([grad[mod1(j, sN)] for j in idx])]
        vl = -dNdt[mod1(lf, sN)] / grad[mod1(lf, sN)] / L
        vr = -dNdt[mod1(rf, sN)] / grad[mod1(rf, sN)] / L
        (; v=(vl + vr) / 2, spread=abs(vl - vr))
    end
end
