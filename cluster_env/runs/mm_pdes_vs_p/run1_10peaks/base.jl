################################################################################
# run1_10peaks - the first MM PDE sweep over p
#
# K=15, l=0.999, m=c=d=1, DN=1e-6, DI=1, and 5 log spaced p = DR/DI from 0.01
# to 1. Each p gets a domain sized by linear stability to hold 10 peaks of its
# fastest growing mode (hence the name) at sN=2500, and 10 repeats started
# from the analytic well-mixed steady state perturbed by +/-1e-3 per gridpoint.
#
# All the machinery lives in ../base.jl; this file is only the parameters of
# this particular sweep. Every value is spelled out below rather than left to
# `make_setup`'s defaults, so that changing those defaults later cannot quietly
# change what this run means.
#
# Results and how to read them: Results.ipynb.
################################################################################
include("../base.jl")

"""
    make_setup1()

Write `setup1.jld2`, the run plan: the well-mixed steady state, the disprel
peak and domain size for each p, and the 50 perturbed initial conditions.

Instant, but it is a separate step on purpose - the u0s are random, so the file
is what pins them down. Generate it once and keep it, and the cluster runs the
same initial conditions as the laptop.
"""
make_setup1() = make_setup("setup1.jld2";
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

"""
    main1()

The full run: all 50 rows of `setup1.jld2`, one thread each, into `main1/`.

`maxtime` is 1h against a measured ~10s for a full-resolution row - the MM is
3 fields, not the 40 of si_pdes1, so the whole sweep is minutes. It is there
only so that a row whose coarsening turns out to be far slower than the ones
measured cannot eat the job; a row that trips it is saved with
`retcode == MaxTime` and needs `overwrite=true` to be retried, since a plain
rerun skips any row that already has a file.

`run_threads` is set for the cluster job's 25 cores (job1.slurm / job1.qsub).
Running this on a laptop is entirely reasonable at this cost - the `min` below
caps it at whatever `-t` gave you.
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
    test_setup1(); test1()

Laptop smoke test of the exact code path main1 takes, downsampled until it
fits in a couple of minutes: sN=250 instead of 2500, 2 p values, 2 reps, and a
60s cap per run so nothing can hang. The dx is 10x coarser than the real runs
so the numbers are not to be trusted - this checks the plumbing (setup file,
solver, per-row output, collection), not the physics.
"""
test_setup1() = make_setup("test_setup1.jld2";
    K=15.0, l=0.999, m=1.0, c=1.0, d=1.0,
    DN=1e-6, DI=1.0,
    ps=10 .^ range(-2, 0, 2),
    num_reps=2,
    sN=250,
    min_peaks=10,
    epsilon=1e-3,
    T=1e8,
    seed=20260820,
)

function test1()
    isfile("test_setup1.jld2") || test_setup1()
    run_pde_setup("test_setup1.jld2", "test1";
        save_step=20,
        maxtime=60,
        run_threads=min(4, nthreads()),
        solver_threads=nothing,
    )
    collect_pde_runs("test1")
end
