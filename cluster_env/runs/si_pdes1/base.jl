using Revise
using SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.RandomSystems

include("../../../scripts/single_influx.jl")

using Base.Threads, OhMyThreads
using Printf
using ProgressMeter
using JLD2, Geppetto
using Random, Distributions
using DataFrames, DataFramesMeta
using LinearAlgebra

# OpenBLAS picks 20 threads on the BIOP nodes, and run_pde_setup already runs
# run_threads solvers at once, so every one of them pulling a 20 thread pool
# inside a --cpus-per-task cgroup is pure contention - the spin-waiting kind.
# Measured on an idle compute-0-6 at the production config (12 cpus, 10 rows,
# solver_threads=nothing), 30s per arm after warming the compile, both orders:
# 1162/1183 accepted steps at 1 thread against 251/260 at 20, ie 4.6x/4.55x.
# UMFPACK's own factorization is single threaded anyway, so nothing here wants
# a BLAS pool.
BLAS.set_num_threads(1)

################################################################################
# ODE solving
################################################################################
"""
    make_stagnation_callback(ps, resid_floor; check_every, t_start)

Terminate with `ReturnCode.Stalled` once the residual stops falling. Every
`check_every` steps we track the smallest `max|du|` seen so far and each time `t`
crosses another decade we give up if that minimum has not dropped 10x and is
still above `resid_floor`. Comparing across decades of `t` rather than fixed
windows keeps slow transients from tripping it, and gating on `resid_floor`
means it can never cut short a run whose residual is already good enough.
Holds mutable state, so make a fresh one per run.
"""
function make_stagnation_callback(ps, resid_floor; check_every=1000, t_start=10.)
    du = zeros(sum(get_Ns(ps)))
    n, tnext, rmin, rprev = 0, t_start, Inf, Inf
    DiscreteCallback(
        function (u, t, _)
            ((n += 1) % check_every == 0 && t > t_start) || return false
            mmicrmfunc!(du, u, ps)
            rmin = min(rmin, maximum(abs, du))
            t < tnext && return false
            stalled = (rmin > 0.1 * rprev) && (rmin > resid_floor)
            rprev, rmin, tnext = rmin, Inf, 10 * tnext
            stalled
        end,
        i -> terminate!(i, ReturnCode.Stalled);
        save_positions=(false, false),
    )
end

################################################################################
# Cluster stage: generate SI systems and find their well-mixed steady states
################################################################################
"""
    solve_si_odes(outfname, num_runs, K, l, T, tol; kwargs...)

Generate `num_runs` random SI systems at the given `K` and `l` and solve the
well-mixed (non-spatial) ODEs to their steady state.

The random system generator still hands out diffusion coefficients but they are
dummies here - they play no part in the non-spatial ODEs - so only the
`mmicrm_params` are kept. Pick the Ds later, at disprel time, with `si_disprel`.

Along with the steady state itself we store `hss_mrl`, the largest real part of
the eigenvalues of the k=0 Jacobian, which says whether the state we landed on
is actually a stable fixed point of the well-mixed dynamics.

Two deliberate differences from si_disprel2, both of which bit there:
- one `extinction_threshold` does both jobs: the exit callback fires when every
  species is below it and `num_surv` counts the ones above it. It defaults to
  `abstol`, which is the level below which the solver makes no promises about a
  component anyway, so calling anything under it extinct is the honest read.
  si_disprel2 used two different values (`abstol/10`, `1e-9`), so the
  solve could stop while species were still above the counting threshold and an
  extinct system would report `num_surv > 0`, with the count depending on which
  step the solver happened to stop on.
- `PositiveDomain` gets `save=false`, otherwise it saves every step it touches
  regardless of `save_everystep=false` (100k+ stored states on stiff systems).
Also storing `final_Ts` so runs that stopped early are easy to spot - anything
with `final_Ts < T` went extinct, stalled or ran out of `maxtime`, and the
retcode says which.
"""
function solve_si_odes(
    outfname, num_runs,
    K, l,
    T, tol;
    abstol=100 * tol,
    reltol=tol,
    N=20, M=20,
    si_u0=[fill(1., N); fill(0., M)],
    solver=TRBDF2,
    maxtime=30.,
    extinction_threshold=abstol,
    stagnation_floor=1e-8, # Inf disables
)
    rsg = get_si_sampler_for_paper(K, l; N, M) # the Ds it makes are dummies, dropped below
    N = rsg.Ns
    metadata = (;
        num_runs, K, l, T, abstol, reltol, N, M, si_u0, solver, maxtime,
        extinction_threshold, stagnation_floor,
        rsg,
    )

    params = Vector{BMMiCRMParams}(undef, num_runs)
    retcodes = Vector{ReturnCode.T}(undef, num_runs)
    final_states = Vector{Vector{Float64}}(undef, num_runs)
    final_Ts = Vector{Float64}(undef, num_runs)
    maxresids = Vector{Float64}(undef, num_runs)
    num_surv = Vector{Int}(undef, num_runs)
    hss_mrls = Vector{Float64}(undef, num_runs)

    prog = Progress(num_runs)
    @tasks for i in 1:num_runs
        si_ps = rsg().mmicrm_params

        si_p = make_mmicrm_problem(si_ps, copy(si_u0), T)
        si_s = solve(si_p, solver();
            dense=false,
            save_everystep=false,
            callback=CallbackSet(make_timer_callback(maxtime), make_ode_extinction_exit_callback(N, extinction_threshold), PositiveDomain(copy(si_u0); save=false), make_stagnation_callback(si_ps, stagnation_floor)),
            abstol=abstol,
            reltol=reltol,
        )
        si_fs = si_s.u[end]

        params[i] = si_ps
        retcodes[i] = si_s.retcode
        final_states[i] = si_fs
        final_Ts[i] = si_s.t[end]
        maxresids[i] = mmicrmmaxresid(si_s)
        num_surv[i] = count(>(extinction_threshold), si_fs[1:N])
        hss_mrls[i] = maximum(real, eigvals!(make_M1(si_ps, si_fs)))

        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)

    df = DataFrame(; params, retcodes, final_states, final_Ts, maxresids, num_surv, hss_mrls)
    mark_good!(df)
    jldsave(outfname; metadata, df)
    df
end

"""
    mark_good!(df; maxresid=1e-8, mrl=1e-9)

Add/refresh the `good` column: the rows worth analysing. A row is good if the
solver actually succeeded, the state really is a steady state (`maxresids`), that
steady state is stable (`hss_mrls`) and something is still alive. Called with the
defaults at the end of `solve_si_odes`, rerun it to use different thresholds.

`ReturnCode.Success` covers both converging and going fully extinct, which is why
the `num_surv` test is needed; `MaxTime` and `Stalled` rows never make it through.
"""
function mark_good!(df; maxresid=1e-8, mrl=1e-9)
    df.good = map(eachrow(df)) do r
        (r.retcodes == ReturnCode.Success) && (r.maxresids < maxresid) &&
            (r.hss_mrls < mrl) && (r.num_surv > 0)
    end
    df
end

################################################################################
# Dispersion relations - the Ds only enter here
################################################################################
"""
    make_si_Ds(ps, DN, DI, DR)

Assemble the diffusion vector matching the state layout ([strains; resources]).
Strains get `DN`, the influx resources get `DI` and all remaining (byproduct)
resources get `DR`. The influx resources are the ones with a non-zero `K`.
"""
function make_si_Ds(ps, DN, DI, DR)
    Ns, Nr = get_Ns(ps)
    Ds = Vector{Float64}(undef, Ns + Nr)
    Ds[1:Ns] .= DN
    for a in 1:Nr
        Ds[Ns+a] = iszero(ps.K[a]) ? DR : DI
    end
    Ds
end

"""
    si_disprels(df, ks, DN, DI, DR; showprogress)

Dispersion relation of every row of a `solve_si_odes` dataframe, evaluated at
`ks`. Returns `(; Ds, mrls)`, the diffusion vector used for each system and its
max real eigenvalue at each k. Stick them on the df with
`df.Ds, df.mrls = si_disprels(df, ks, DN, DI, DR)`.
"""
function si_disprels(df, ks, DN, DI, DR;
    showprogress=true,
)
    Ds = Vector{Vector{Float64}}(undef, nrow(df))
    mrls = Vector{Vector{Float64}}(undef, nrow(df))
    prog = Progress(nrow(df); enabled=showprogress)
    @tasks for i in 1:nrow(df)
        Ds[i] = make_si_Ds(df.params[i], DN, DI, DR)
        mrls[i] = linstab_simple(df.params[i], Ds[i], df.final_states[i], ks)
        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)
    (; Ds, mrls)
end

################################################################################
# Peak wavenumbers and picking a system size
################################################################################
"""
    disprel_peak_ks(df, ks, DN, DI, ps; showprogress)

Dispersion relation of every row of `df` at each `DR` (`p`) in `ps`, plus where
each one peaks. Only the `ks` grid is searched, no root finding, so a peak k is
only good to the grid spacing - which is all the system size picking below
needs, since that gets rounded to a whole number of peaks anyway.

Returns `(; mrls, peak_ks, peak_hs, at_edge)`, each a `nrow(df) x length(ps)`
matrix. A system whose disprel never gets above zero is spatially stable at that
p and gets `missing` for its peak. `at_edge` marks the peaks sitting on the first
or last k, where the real maximum is outside the scanned range and the k should
not be trusted - widen `ks` if any show up.

Pass `df[df.good, :]`, not the whole df: rows that never reached a stable steady
state have nothing to be perturbed around and their disprels are meaningless.
"""
function disprel_peak_ks(df, ks, DN, DI, ps;
    showprogress=true,
)
    n = nrow(df)
    mrls = Matrix{Vector{Float64}}(undef, n, length(ps))
    peak_ks = Matrix{Union{Missing,Float64}}(undef, n, length(ps))
    peak_hs = Matrix{Union{Missing,Float64}}(undef, n, length(ps))
    at_edge = falses(n, length(ps))

    for (j, p) in enumerate(ps)
        mrls[:, j] = si_disprels(df, ks, DN, DI, p; showprogress).mrls
        for i in 1:n
            h, ki = findmax(mrls[i, j])
            if h <= 0.0
                peak_ks[i, j] = missing
                peak_hs[i, j] = missing
            else
                peak_ks[i, j] = ks[ki]
                peak_hs[i, j] = h
                at_edge[i, j] = (ki == 1) || (ki == length(ks))
            end
        end
    end

    (; mrls, peak_ks, peak_hs, at_edge)
end

"""
    num_peaks_in(L, k)

How many peaks a mode of wavenumber `k` puts in a periodic domain of length `L`.
The one place the peaks <-> wavenumber convention lives, so `choose_Ls` and any
check of what it returned cannot disagree about a factor of 2pi.
"""
num_peaks_in(L, k) = L * k / (2pi)

"""
    choose_L(k, num_peaks)

Inverse of [`num_peaks_in`](@ref): the domain length that fits exactly
`num_peaks` peaks of wavenumber `k`.
"""
choose_L(k, num_peaks) = 2pi * num_peaks / k

"""
    choose_Ls(peak_ks, min_peaks)

Pick one domain size per system from its peak wavenumbers across the ps, big
enough to fit at least `min_peaks` peaks *at every p*. Inverting `num_peaks_in`
gives `L = 2pi * min_peaks / k`, so it is the smallest peak k (the longest
wavelength, generally the smallest p) that sets the size and the other ps get
proportionally more peaks.

`peak_ks` is the matrix from `disprel_peak_ks`; the ps where a system is stable
are `missing` and simply do not constrain it. Returns `(; Ls, num_unstable)`,
with `Ls` `missing` for a system that is stable at every p and so has no
lengthscale to fit.
"""
function choose_Ls(peak_ks, min_peaks)
    Ls = Vector{Union{Missing,Float64}}(undef, size(peak_ks, 1))
    num_unstable = Vector{Int}(undef, size(peak_ks, 1))
    for i in axes(peak_ks, 1)
        pks = collect(skipmissing(peak_ks[i, :]))
        num_unstable[i] = length(pks)
        Ls[i] = isempty(pks) ? missing : choose_L(minimum(pks), min_peaks)
    end
    (; Ls, num_unstable)
end

################################################################################
# PDE solving
################################################################################
"""
    run_si_pde(ps, Ds, u0, T, L; kwargs...)

Solve one 1d SI PDE on a periodic domain of length `L` with `size(u0, 2)`
gridpoints, from `u0` to time `T`.

Takes `ps` (a `BMMiCRMParams`) and `Ds` separately since that is how the setup
file stores them - the `BSMMiCRMParams` gluing them to a space is built here,
where `dx = L / sN` is known.

Solver setup is the one from single_influx_pdes6's later runs (main4/main5):
`QNDF` with a loose `abstol` and a tight `reltol`, which was the combination
that got these systems to their steady state in a sane amount of wall clock
without the solver wandering negative, plus `PositiveDomain` on top of it. The
tolerances are the knobs worth turning, so they are kwargs; `save=false` on
`PositiveDomain` keeps it from saving every step it touches.

Returns `(; s, saved_ts, saved_us)` - the solution plus a series of states over
time for watching the coarsening, every `save_step`-th solver step via
`make_stepped_saver_callback`, same as si_totbiom. Saving actual steps costs
nothing and the states are exactly what the solver produced, `PositiveDomain`
included. The price is an irregular time grid that differs between runs and a
state count that rides on how many steps the solver takes: si_totbiom's runs
gave anywhere from 17 to 344 at `save_step=200`. At 800KB a state here, budget
for that - 1250 states would be a 1GB row file.

`save_step=nothing` drops the saver callback entirely and returns empty
`saved_ts`/`saved_us`, for runs where only the final state is wanted.
"""
function run_si_pde(ps, Ds, u0, T, L;
    abstol=1e-7,
    reltol=1e-9,
    solver=QNDF,
    maxtime=60 * 60,
    solver_threads=nothing,
    save_step=500,
    kwargs...
)
    sN = size(u0, 2)
    dx = L / sN

    sps = BSMMiCRMParams(ps, Ds, CartesianSpace{1,Tuple{Periodic}}(SA[dx]), solver_threads)
    sp = make_smmicrm_problem(sps, copy(u0), T; jac_type=:sparse)

    if isnothing(save_step)
        saved_ts, saved_us, save_cbs = Float64[], typeof(u0)[], ()
    else
        saved_ts, saved_us, scb = make_stepped_saver_callback(u0, save_step)
        push!(saved_ts, 0.0)
        push!(saved_us, copy(u0))
        save_cbs = (scb,)
    end

    s = solve(sp, solver();
        dense=false,
        save_everystep=false,
        calck=false,
        abstol,
        reltol,
        callback=CallbackSet(make_timer_callback(maxtime), PositiveDomain(copy(u0); save=false), save_cbs...),
        kwargs...
    )

    if !isnothing(save_step)
        push!(saved_ts, s.t[end])
        push!(saved_us, s.u[end])
    end

    (; s, saved_ts, saved_us)
end

"""
    run_pde_setup(setup_fname, outdir; kwargs...)

Run every row of the `pde_df` in a setup file (the jld2 the notebook stage
writes: one row per (system, p), each carrying its `params`, `Ds`, `L`, `sN`,
`T` and `u0`) and write one small jld2 per row into `outdir`.

Per-row files rather than one big one because these runs are hours each: a job
that hits its walltime keeps everything it finished, and rerunning it picks up
where it stopped - rows whose file already exists are skipped unless
`overwrite=true`. Use `rows` to split the work across array jobs. A row that
throws is reported and left without a file, so a rerun retries it.

`run_threads` rows are solved at once, each solver using `solver_threads`
threads. With more rows than cores, one row per thread and a serial solver is
what scaled in single_influx_pdes6; with fewer, give each solver the leftover
threads. Set both in the calling `main`, where the row count is known.

Read the results back with `collect_pde_runs`.
"""
function run_pde_setup(setup_fname, outdir;
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
        setup_fname, abstol, reltol, solver, maxtime, run_threads, solver_threads,
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
        @set scheduler = :greedy
        @set ntasks = run_threads

        r = pde_df[i, :]
        try
            start_time = time()
            (; s, saved_ts, saved_us) = run_si_pde(r.params, r.Ds, r.u0, r.T, r.L;
                abstol, reltol, solver, maxtime, solver_threads, kwargs...
            )
            realtime = time() - start_time

            jldsave(joinpath(outdir, @sprintf("row%04d.jld2", i));
                pde_df_row=i, gdf_row=r.gdf_row, df_row=r.df_row, p=r.p,
                L=r.L, sN=r.sN, T=r.T,
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
`pde_df_row`. The heavy `params`/`Ds`/`u0` stay in the setup file - join on
`pde_df_row` when they are needed.

The `saved_us` series is most of a row file, so it is left on disk unless you ask for
`states=true`; `load_pde_states` gets one row's worth, which is what plotting
actually wants. Reading is per-key so the rest of a row costs nothing.
"""
function collect_pde_runs(outdir; states=false)
    cols = ["pde_df_row", "gdf_row", "df_row", "p", "L", "sN", "T",
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

The saved states of one row as `(; ts, us)`, `us[k]` being the full 40 x sN
state at `ts[k]`. Row `i` is the `pde_df_row`, ie the index into the setup
file's `pde_df`, not a position in `collect_pde_runs`.
"""
function load_pde_states(outdir, i)
    jldopen(joinpath(outdir, @sprintf("row%04d.jld2", i))) do d
        (; ts=d["saved_ts"], us=d["saved_us"])
    end
end
