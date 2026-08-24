################################################################################
# run2_100peaks - run1 with 10x the domain, to test whether the arrest is real
#
# Identical to run1_10peaks in every parameter but one: each domain is sized to
# hold 100 peaks of its fastest growing mode instead of 10. Everything else -
# K=15, l=0.999, m=c=d=1, DN=1e-6, DI=1, the same 5 log spaced p, 10 repeats,
# sN=2500, the same perturbation and solver settings - is unchanged.
#
# The question it answers: run1 coarsened from 10 peaks and stopped at 3-6,
# never reaching one. That could be a real arrest at a preferred spacing, or it
# could be the box running out of room. If the spacing is what is preferred,
# the final count here should be ~10x run1's; if the count is, it should be the
# same 3-6.
#
# THE CATCH, and it is a big one: sN is held at 2500 while L grows 10x, so dx
# grows 10x too. That is 25 gridpoints per initial wavelength, against run1's
# 250. See Results.ipynb for what that does to the final spikes - these numbers
# should not be trusted until sN goes up.
#
# All the machinery lives in ../base.jl; this file is only this sweep's
# parameters.
################################################################################
include("../base.jl")

"""
    make_setup1()

Write `setup1.jld2`, the run plan: the well-mixed steady state, the disprel
peak and domain size for each p, and the 50 perturbed initial conditions.

Same as run1's but with `min_peaks=100`, and a different `seed` so the repeats
are independent draws rather than run1's noise field on a different domain.
"""
make_setup1() = make_setup("setup1.jld2";
    K=15.0, l=0.999, m=1.0, c=1.0, d=1.0,
    DN=1e-6, DI=1.0,
    ps=10 .^ range(-2, 0, 5),
    num_reps=10,
    sN=2500,
    min_peaks=100,
    epsilon=1e-3,
    T=1e8,
    ks=10 .^ range(-6, 4, 20001),
    seed=20260821,
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
        save_step=20,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=60 * 60,
        run_threads=min(25, nthreads()),
        solver_threads=nothing,
    )
end

"""
    res_check1(; p_i, rep, sNs, outdir)

The check run2 needs: is the arrested peak count converged in `dx`, or is it an
artifact of holding sN=2500 while L grew 10x?

Re-solves one (p, rep) of `setup1.jld2` at several `sNs` on the same domain, so
the only thing changing is the grid. Defaults to `p_i=1` (p=0.01), which has
the tightest final spacing in gridpoints and is therefore the worst case.

The initial condition is generated once on the coarsest grid and linearly
interpolated onto the finer ones, so every run starts from the *same physical*
perturbation. Drawing fresh noise per grid would confound resolution
differences with ordinary run-to-run scatter, which at p=0.01 is +/-4 peaks.
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
