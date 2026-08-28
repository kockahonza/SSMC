################################################################################
# run7_final_100peaks_DN1e-3 - the final version of the 100 peak DN=1e-3 sweep
#
# run3 with sN doubled to 5000, 12 reps instead of 10, and p=2 added on top of
# the 5 log spaced p in [0.1, 1]. K=5, l=0.999, m=c=d=1, DN=1e-3, DI=1, 100 peaks
# and the same perturbation size and solver settings are as they were.
#
# DN=1e-3 is the point of the whole family: it is the one parameter that can
# plausibly unstick the coarsening arrest run1 and run2 found, since it is what
# lets biomass move between peaks rather than only through the resource fields.
# K is what has to move with it - at K=15 the instability dies around DN ~ 8e-5
# (p=0.01) to 5e-6 (p=1), so DN=1e-3 sits 1-2 decades inside the stable region
# and there is no pattern to coarsen at all. Lowering K lowers the well-mixed N*,
# which raises the gain and lengthens the diffusion lengths enough to reopen the
# band, and it is a narrow window: at DN=1e-3 every p up to 2 is unstable only
# for K between ~3 (below which N*=0 and there is no biomass) and ~5.x. K=6
# already loses p>=0.56 and K=10 is stable everywhere. Hence K=5, as in run3.
#
# sN=5000 is the other fix. run3 kept run2's sN=2500 on a ~4x larger domain, so
# its dx was ~4x coarser than run2's, which was itself short of its own
# convergence test; res_check1 there measured the cost. Doubling sN halves dx.
#
# All the machinery lives in ../base.jl; this file is only this sweep's
# parameters.
################################################################################
include("../base.jl")

"""
    make_setup1()

Write `setup1.jld2`, the run plan: the well-mixed steady state, the disprel
peak and domain size for each p, and the 72 perturbed initial conditions.

`K=5` and `DN=1e-3` are the pair that has to move together - see the header.
`make_setup` refuses to write a setup whose ps are not all spatially unstable,
and that guard is what set both ends of the p range: p=10 and p=3 come back
STABLE at this K and DN, so p=2 (peak_h=0.0026, 33x weaker than p=0.1) is the
last one in.

The `seed` differs from run2's on purpose. sN and epsilon are unchanged, so
reusing run2's seed would have handed every rep the byte-identical noise field
laid on a different steady state - a paired comparison rather than an
independent one.
"""
make_setup1() = make_setup("setup1.jld2";
    K=5.0, l=0.999, m=1.0, c=1.0, d=1.0,
    DN=1e-3, DI=1.0,
    ps=[10 .^ range(-1, 0, 5); 2],
    num_reps=12,
    sN=5000,
    min_peaks=100,
    epsilon=1e-3,
    T=1e12,
    ks=10 .^ range(-5, 3, 20000),
    seed=20260823,
)

"""
    main1()

The full run: all 72 rows of `setup1.jld2`, one thread each, into `main1/`.

Same solver settings as run1. Coarsening from 100 peaks takes many more merge
events than from 10, so these runs are longer; `maxtime` caps a row at 1h and
marks it `retcode == MaxTime`, which a plain rerun will skip - pass
`overwrite=true` to retry those.
"""
function main1()
    run_pde_setup("setup1.jld2", "main1";
        save_step=20,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=60 * 60,
        run_threads=nthreads(),
        solver_threads=nothing,
    )
end

"""
    res_check1(; p_i, rep, sNs, outdir)

Is the arrested peak count converged in `dx`, or an artifact of the grid?

Re-solves one (p, rep) of `setup1.jld2` at several `sNs` on the same domain, so
the only thing changing is the grid. Defaults to `p_i=1` (p=0.1), which has
the shortest byproduct diffusion length sqrt(DR/v) and so the tightest
structure in gridpoints - the worst case.

This matters more here than it did in run2. run2 needed ~5.5 gridpoints per
sqrt(DR/v) to converge, and its sN=2500 gave 2.8 and overcounted peaks by ~9%;
run3's ~4x larger L puts sN=2500 nearer 1.3, so expect the coarsest grid to be
off and convergence to want a larger sN than run2 needed.

The initial condition is generated once on the coarsest grid and linearly
interpolated onto the finer ones, so every run starts from the *same physical*
perturbation. Drawing fresh noise per grid would confound resolution against
ordinary run-to-run scatter.
"""
function res_check1(;
    p_i=1, rep=1, sNs=[2500, 5000, 10000, 20000],
    outdir="res_check1", maxtime=3*60*60, seed=20260822,
)
    setup = load("setup1.jld2")
    md, pde_df = setup["metadata"], setup["pde_df"]
    r = only(eachrow(pde_df[(pde_df.p_i .== p_i) .& (pde_df.rep .== rep), :]))
    mkpath(outdir)

    # one perturbation field on the coarsest grid, resampled onto the others
    base_sN = minimum(sNs)
    Random.seed!(seed)
    base_u0 = clamp!(perturb_u0_uniform(1, 2,
        expand_u0_to_size((base_sN,), md.hss), md.epsilon), 0.0, Inf)

    function resample(u, sN)
        out = Matrix{Float64}(undef, size(u, 1), sN)
        for i in 1:sN
            x = (i - 0.5) / sN * base_sN + 0.5      # position in base grid units
            i0 = floor(Int, x); w = x - i0
            a, b = mod1(i0, base_sN), mod1(i0 + 1, base_sN)
            @views out[:, i] .= (1 - w) .* u[:, a] .+ w .* u[:, b]
        end
        out
    end

    @printf("res_check1: p=%.4g rep=%d, L=%.4f, sNs=%s\n", r.p, rep, r.L, string(sNs))
    flush(stdout)

    @tasks for i in eachindex(sNs)
        @set ntasks = length(sNs)
        sN = sNs[i]
        u0 = sN == base_sN ? copy(base_u0) : resample(base_u0, sN)
        t0 = time()
        (; s, saved_ts, saved_us) = run_mm_pde(r.mmicrm_params, r.Ds, u0, r.T, r.L;
            maxtime, save_step=50, solver_threads=nothing)
        jldsave(joinpath(outdir, @sprintf("sN%06d.jld2", sN));
            sN, p=r.p, rep, L=r.L, T=r.T,
            retcode=s.retcode, final_state=s.u[end], final_T=s.t[end],
            maxresid=smmicrmmaxresid(s), realtime=time() - t0,
            saved_ts, saved_us)
        @printf("  sN=%d done: %s, %.0fs\n", sN, string(s.retcode), time() - t0)
        flush(stdout)
    end
    outdir
end
