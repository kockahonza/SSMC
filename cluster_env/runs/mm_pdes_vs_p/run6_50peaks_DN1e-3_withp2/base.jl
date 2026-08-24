################################################################################
# run6_50peaks_DN1e-3_withp2 - run5 plus p=2
#
# Same sweep as run5 - K=5, DN=1e-3, l=0.999, m=c=d=1, DI=1, 50 peaks per
# domain, sN=5000, 10 reps, T=1e12, abstol=1e-7, reltol=1e-9 - with p=2 added
# to the five log spaced values, so 6 p and 60 rows.
#
# p=2 is close to the stability boundary at this DN: peak growth rate 2.58e-3
# against 1.29e-2 at p=1, and the critical DN there is 1.26e-3 against the 1e-3
# used here. p=3 is already stable. Expect those rows to start late and to be
# the slowest in the sweep, which is why maxtime is raised below.
#
# Built to run as a cluster array job, one row per task - see job_array.qsub
# (Myriad, SGE) and job_array.slurm (dias, Slurm), and main1_row.
#
# All the machinery lives in ../base.jl; this file is only this sweep's
# parameters.
################################################################################
include("../base.jl")

"""
    make_setup1()

Write `setup1.jld2`, the run plan: the well-mixed steady state, the disprel
peak and domain size for each p, and the 50 perturbed initial conditions.

`ps` gains 2.0 on top of run5's five log spaced values; everything else -
K=5, DN=1e-3, min_peaks=50, sN=5000 and the tolerances - is run5's. `make_setup` refuses to write a setup whose
ps are not all spatially unstable, so running this is also the cheap test of
whether K=5 really does reopen the band at this DN.

The `seed` differs from run5's on purpose. sN and epsilon are unchanged, so
reusing run2's seed would have handed every rep the byte-identical noise field
laid on a different steady state - a paired comparison rather than an
independent one.
"""
make_setup1() = make_setup("setup1.jld2";
    K=5.0, l=0.999, m=1.0, c=1.0, d=1.0,
    DN=1e-3, DI=1.0,
    ps=[10 .^ range(-2, 0, 5); 2.0],
    num_reps=10,
    sN=5000,
    min_peaks=50,
    epsilon=1e-3,
    T=1e12,
    ks=10 .^ range(-6, 4, 20001),
    seed=20260826,
)

"""
    main1()

The full run: all 50 rows of `setup1.jld2`, one thread each, into `main1/`.

Same solver settings as run1. Coarsening from 100 peaks takes many more merge
events than from 10, so these runs are longer; `maxtime` caps a row at 1h and
marks it `retcode == MaxTime`, which a plain rerun will skip - pass
`overwrite=true` to retry those.
"""
function main1()
    run_pde_setup("setup1.jld2", "main1";
        # storage only - it changes nothing the solver does, just how often a
        # state is kept. At sN=5000 a state is 120KB, and the p=1 rows take
        # ~300k steps, so save_step=20 would hold ~1.8GB per row and 10 of
        # those at once is what put this laptop into swap once already.
        save_step=200,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=6 * 60 * 60,   # p=2 rows are slow, see header
        run_threads=min(25, nthreads()),
        solver_threads=nothing,
    )
end

"""
    res_check1(; p_i, rep, sNs, outdir)

Is the arrested peak count converged in `dx`, or an artifact of the grid?

Re-solves one (p, rep) of `setup1.jld2` at several `sNs` on the same domain, so
the only thing changing is the grid. Defaults to `p_i=1` (p=0.01), which has
the shortest byproduct diffusion length sqrt(DR/v) and so the tightest
structure in gridpoints - the worst case.

This matters more here than it did in run2. run2 needed ~5.5 gridpoints per
sqrt(DR/v) to converge, and its sN=2500 gave 2.8 and overcounted peaks by ~9%;
run4's L is 5x smaller than run3's at the same sN, so dx is 5x finer and every
p has more margin than run3 - which was itself already converged at p=0.01.

The initial condition is generated once on the coarsest grid and linearly
interpolated onto the finer ones, so every run starts from the *same physical*
perturbation. Drawing fresh noise per grid would confound resolution against
ordinary run-to-run scatter.
"""
function res_check1(;
    p_i=1, rep=1, sNs=[5000, 10000, 20000],
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

"""
    main1_row(i)

One row of `setup1.jld2` into `main1/`, single threaded - the entry point for
the cluster array jobs, which give each task one core and one row.

Rows that already have a file are skipped, so a resubmit of the whole array
only picks up what is missing.
"""
main1_row(i) = run_pde_setup("setup1.jld2", "main1";
    rows=[i], run_threads=1, solver_threads=nothing,
    save_step=200, abstol=1e-7, reltol=1e-9, maxtime=6 * 60 * 60,
)
