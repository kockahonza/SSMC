################################################################################
# run4_arrest_check - is the arrest at p=1, DN=1e-6 a real steady state?
#
# run2 reported every run "arrested": the peak count stopped changing and sat
# still for hundreds of times longer than it took to get there. That is evidence
# but not proof. Three things can look exactly like an arrest and are not:
#
#   1. a state pinned to the grid, which unpins when dx shrinks;
#   2. a state frozen by the error control, which moves when the tolerances drop;
#   3. coarsening that is merely exponentially slow, which resumes if you watch
#      for longer than T = 1e8.
#
# Nothing that starts from noise can separate these, because every such run
# conflates "would the system get here" with "does it stay here". So this run
# does it the other way round: solve **one** system once, take its final state,
# and restart *from that state* under conditions that each of the three failure
# modes would react to. Same state, different numerics - any change in the peak
# count is then attributable to the numerics and nothing else.
#
# One p only. p=1 with DN=1e-6 is where run2's arrest is least ambiguous
# (26-29 peaks from 100) and where the /tmp four-variant test reproduced 29,
# so it is the case worth pinning down first.
#
# sN=5000 rather than run2's 2500: res_check1 showed 2500 is under-resolved
# (1.6 gridpoints across a spike) and biased towards *less* coarsening, which is
# the direction that would fake an arrest. 5000 is the coarsest grid res_check1
# found converged.
#
# Stage 1 `solve1`   - the one long solve, T=1e8, writes state1.jld2
# Stage 2 `battery1` - the restarts, writes restarts/*.jld2
#         `report1`  - the summary table
#
# The shared machinery is ../base.jl; only the pieces that need a switch it does
# not have (positivity) or that are specific to restarting are defined here.
################################################################################
include("../base.jl")

# qualified: Interpolations exports `Periodic`, `bounds` and `scale`, all of which
# collide with names SSMCMain needs unshadowed - `Periodic` is the space's
# boundary condition type and is used a few lines down in every solve.
import Interpolations as Itp

const P_TARGET = 1.0
const SN_BASE = 5000

################################################################################
# Stage 1: the one solve whose final state everything else starts from
################################################################################
"""
    make_setup1()

Write `setup1.jld2`: one p, one rep, `sN=5000`, a domain sized for 100 peaks.

Uses run2's parameters unchanged (`K=15`, `l=0.999`, `m=c=d=1`, `DN=1e-6`,
`DI=1`) so the state this produces is the same object run2 was reporting on,
not a nearby one. `ps=[1.0]` because `make_setup` takes a vector; the linear
stability table it prints is then a single row.

A different `seed` from run2's, since there is no reason to want run2's
particular noise field and a fresh one makes the two runs independent samples
of the same distribution.
"""
make_setup1() = make_setup("setup1.jld2";
    K=15.0, l=0.999, m=1.0, c=1.0, d=1.0,
    DN=1e-6, DI=1.0,
    ps=[P_TARGET],
    num_reps=1,
    sN=SN_BASE,
    min_peaks=100,
    epsilon=1e-3,
    T=1e8,
    ks=10 .^ range(-6, 4, 20001),
    seed=20260821,
)

"""
    run_mm_pde2(mmicrm_params, Ds, u0, T, L; positivity, ...)

`run_mm_pde` from ../base.jl with two things it does not have: the positivity
projection made optional, and a `t0` so a restart can be labelled with the time
it is continuing from.

`PositiveDomain` is hardcoded in `run_mm_pde`, and one of the three failure
modes this run is testing is whether it matters, so the switch has to exist
somewhere. It lives here rather than in ../base.jl so that run1-run3 keep
solving exactly what they solved.
"""
function run_mm_pde2(mmicrm_params, Ds, u0, T, L;
    positivity=true,
    abstol=1e-7,
    reltol=1e-9,
    solver=QNDF,
    maxtime=60 * 60,
    solver_threads=nothing,
    save_step=20,
    t0=0.0,
    kwargs...
)
    sN = size(u0, 2)
    dx = L / sN

    sps = BSMMiCRMParams(mmicrm_params, Ds,
        CartesianSpace{1,Tuple{Periodic}}(SA[dx]), solver_threads)
    sp = make_smmicrm_problem(sps, copy(u0), T; jac_type=:sparse, t0)

    saved_ts, saved_us, scb = make_stepped_saver_callback(u0, save_step)
    push!(saved_ts, t0)
    push!(saved_us, copy(u0))

    cbs = Any[make_timer_callback(maxtime), scb]
    positivity && insert!(cbs, 2, PositiveDomain(copy(u0); save=false))

    s = solve(sp, solver();
        dense=false, save_everystep=false, calck=false,
        abstol, reltol, callback=CallbackSet(cbs...), kwargs...)

    push!(saved_ts, s.t[end])
    push!(saved_us, s.u[end])

    (; s, saved_ts, saved_us, sps)
end

"""
    solve1(; maxtime)

Stage 1. Solve the single row of `setup1.jld2` with run2's production settings
(`PositiveDomain` on, `abstol=1e-7`, `reltol=1e-9`, `QNDF`) and write
`state1.jld2`.

Deliberately the production settings and not something better: the point of the
restarts is to interrogate *the state run2's numerics produce*, so this has to
produce it the same way. Whether those settings are good enough is what stage 2
is for.
"""
function solve1(; maxtime=6 * 60 * 60, save_step=20)
    setup = load("setup1.jld2")
    r = only(eachrow(setup["pde_df"]))

    @printf("solve1: p=%.4g L=%.4f sN=%d dx=%.6g lam*=%.4f (%.1f pts/lam) T=%.3g\n",
        r.p, r.L, r.sN, r.L / r.sN, 2pi / r.peak_k, (2pi / r.peak_k) / (r.L / r.sN), r.T)
    flush(stdout)

    t0 = time()
    (; s, saved_ts, saved_us, sps) = run_mm_pde2(r.mmicrm_params, r.Ds, r.u0, r.T, r.L;
        positivity=true, abstol=1e-7, reltol=1e-9, maxtime, save_step)
    realtime = time() - t0

    u_star = s.u[end]
    np = count_peaks(u_star[1, :])
    resid = smmicrmmaxresid(u_star, sps)

    jldsave("state1.jld2";
        p=r.p, L=r.L, sN=r.sN, T=r.T, peak_k=r.peak_k,
        mmicrm_params=r.mmicrm_params, Ds=r.Ds, u0=r.u0,
        retcode=s.retcode, final_T=s.t[end], u_star, np, resid, realtime,
        saved_ts, saved_us, setup_metadata=setup["metadata"])

    @printf("  %s  final_T=%.4g  peaks=%d  maxresid=%.3g  max_N=%.5g  %.0fs\n",
        string(s.retcode), s.t[end], np, resid, maximum(u_star[1, :]), realtime)
    flush(stdout)
    (; u_star, np, resid)
end

################################################################################
# Stage 2: restarting from that state
################################################################################
"""
    resample_periodic(u, sN_new)

A `(3, sN)` periodic state on cell centres, resampled onto `sN_new` cell
centres with a periodic cubic B-spline.

Cubic rather than the linear interpolation res_check1 used, because here the
interpolated state is fed straight back into the *residual*, and linear
interpolation puts a kink in every cell that the second-derivative operator
turns into a large spurious residual. Cubic is C2, so what the Laplacian sees
is the field and not the interpolant's corners.

Even so, the interpolation is not exact, and its error shows up as a residual
that has nothing to do with whether the state is a steady state. That is why
`run_restart` reports the residual both immediately after resampling and again
after the solve: the first is contaminated, the second is not.
"""
function resample_periodic(u, sN_new)
    sN = size(u, 2)
    sN == sN_new && return copy(u)
    out = Matrix{Float64}(undef, size(u, 1), sN_new)
    for c in axes(u, 1)
        itp = Itp.extrapolate(
            Itp.interpolate(u[c, :], Itp.BSpline(Itp.Cubic(Itp.Periodic(Itp.OnCell())))),
            Itp.Periodic())
        for j in 1:sN_new
            # cell centre j of the new grid, in old-grid index units
            out[c, j] = itp((j - 0.5) * sN / sN_new + 0.5)
        end
    end
    out
end

"""
    run_restart(st, name; sN, abstol, reltol, positivity, T, perturb, seed, maxtime)

One restart from `state1.jld2`'s `u_star`, written to `restarts/<name>.jld2`.

`st` is the loaded `state1.jld2`. The state is resampled to `sN` (a no-op at the
default), optionally perturbed, then integrated to `T` under the given solver
settings. What comes back is the peak count and residual before and after, so a
restart that moved can be told from one that did not.

`perturb` is a *relative* uniform perturbation, applied to every component and
gridpoint. It is the test the other knobs cannot do: a state can be a fixed
point of the discretisation and still be one the dynamics would leave under the
slightest nudge, which is what a marginally-pinned state looks like.
"""
function run_restart(st, name;
    sN=st["sN"], abstol=1e-7, reltol=1e-9, positivity=true,
    T=1e8, perturb=0.0, seed=20260822, maxtime=6 * 60 * 60, save_step=20,
)
    mkpath("restarts")
    u_in = resample_periodic(st["u_star"], sN)
    if perturb > 0
        Random.seed!(seed)
        u_in .*= 1 .+ perturb .* (2 .* rand(size(u_in)...) .- 1)
        clamp!(u_in, 0.0, Inf)
    end

    L = st["L"]
    sps_in = BSMMiCRMParams(st["mmicrm_params"], st["Ds"],
        CartesianSpace{1,Tuple{Periodic}}(SA[L / sN]), nothing)
    np_in = count_peaks(u_in[1, :])
    resid_in = smmicrmmaxresid(u_in, sps_in)

    t0 = time()
    (; s, saved_ts, saved_us, sps) = run_mm_pde2(st["mmicrm_params"], st["Ds"],
        u_in, T, L; positivity, abstol, reltol, maxtime, save_step)
    realtime = time() - t0

    u_out = s.u[end]
    np_out = count_peaks(u_out[1, :])
    resid_out = smmicrmmaxresid(u_out, sps)

    jldsave(joinpath("restarts", "$(name).jld2");
        name, sN, abstol, reltol, positivity, T, perturb,
        retcode=s.retcode, final_T=s.t[end],
        np_in, np_out, resid_in, resid_out, u_out, realtime,
        saved_ts, saved_us)

    @printf("%-16s sN=%-6d atol=%-8.3g rtol=%-8.3g pos=%-5s T=%-8.3g pert=%-8.3g | %-8s peaks %3d -> %3d  resid %9.3g -> %9.3g  %.0fs\n",
        name, sN, abstol, reltol, string(positivity), T, perturb,
        string(s.retcode), np_in, np_out, resid_in, resid_out, realtime)
    flush(stdout)
    nothing
end

"""
    battery1(; maxtime)

Stage 2. Every restart, each isolating one of the three ways an arrest can be
fake.

- `control`      - same grid, same settings, another 1e8 of time. If this moves,
                   nothing else means anything.
- `fine_*`       - the grid refined 2x, 4x and 8x. Answers failure mode 1: a
                   pinned state unpins when dx shrinks.
- `tight`        - abstol and reltol at 1e-12. Answers failure mode 2.
- `nopos`        - no positivity projection. Also failure mode 2, and the thing
                   the /tmp test cleared at 2500 but not yet at this grid.
- `long_T`       - to 1e12, ten thousand times longer. Answers failure mode 3:
                   coarsening that is merely very slow.
- `fine_long`    - refined *and* long, because the two can hide each other: a
                   fine grid unpins the state but 1e8 may be too short to see
                   the result, and a long run on a pinned grid cannot move at
                   all.
- `perturb_*`    - nudged by 1e-6 and 1e-3 relative. Tests whether the state is
                   dynamically stable or merely sitting still.
"""
function battery1(; maxtime=6 * 60 * 60)
    st = load("state1.jld2")
    @printf("\nbattery1: restarting from state1.jld2 - %d peaks, resid %.3g, L=%.4f\n\n",
        st["np"], st["resid"], st["L"])
    flush(stdout)

    run_restart(st, "control")
    run_restart(st, "fine_10000"; sN=10000)
    run_restart(st, "fine_20000"; sN=20000)
    run_restart(st, "fine_40000"; sN=40000, maxtime)
    run_restart(st, "tight"; abstol=1e-12, reltol=1e-12)
    run_restart(st, "nopos"; positivity=false)
    run_restart(st, "long_T"; T=1e12)
    run_restart(st, "fine_long"; sN=20000, T=1e12, maxtime)
    run_restart(st, "perturb_1e-6"; perturb=1e-6)
    run_restart(st, "perturb_1e-3"; perturb=1e-3)
    nothing
end

"""
    report1()

The restarts as one table, sorted the way they should be read: the control
first, then each failure mode.

`moved` is the column that matters - whether the peak count changed. A run that
did not move under a knob has cleared the failure mode that knob tests.
"""
function report1()
    st = load("state1.jld2")
    fs = filter(f -> endswith(f, ".jld2"), readdir("restarts"))
    rows = map(fs) do f
        d = load(joinpath("restarts", f))
        (; name=d["name"], sN=d["sN"], abstol=d["abstol"], reltol=d["reltol"],
            pos=d["positivity"], T=d["T"], perturb=d["perturb"],
            np_in=d["np_in"], np_out=d["np_out"],
            moved=d["np_out"] != d["np_in"],
            resid_in=d["resid_in"], resid_out=d["resid_out"],
            retcode=string(d["retcode"]), final_T=d["final_T"],
            realtime=d["realtime"])
    end
    df = DataFrame(rows)
    order = ["control", "fine_10000", "fine_20000", "fine_40000", "tight", "nopos",
        "long_T", "fine_long", "perturb_1e-6", "perturb_1e-3"]
    df.ord = [something(findfirst(==(n), order), 99) for n in df.name]
    sort!(df, :ord)
    select!(df, Not(:ord))

    @printf("state1: %d peaks at sN=%d, resid %.3g\n", st["np"], st["sN"], st["resid"])
    @printf("restarts that changed the peak count: %d / %d\n\n",
        count(df.moved), nrow(df))
    df
end
