################################################################################
# For Fig.3e, multiple systems each solved at 5 different p
################################################################################
include("../base.jl")
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

"""
    part_rows(setup_fname, part, nparts)

The rows of a setup file's `pde_df` that job `part` of `nparts` should run, ie
what to hand `run_pde_setup`'s `rows`, so one setup file can be worked through
by several jobs on different nodes.

Deals the *systems* out round robin rather than cutting the row range in pieces,
so all of a system's ps stay in the same job and the parts differ by at most one
system. Since every system contributes the same number of rows, that is an even
split of the rows too.

The indices are positions in the full `pde_df`, which is what `run_pde_setup`
names its output files by - so the parts never collide, can share one outdir,
and `collect_pde_runs`/`load_pde_states` need to know nothing about the split.
"""
function part_rows(setup_fname, part, nparts)
    pde_df = load(setup_fname, "pde_df")
    syss = Set(unique(pde_df.gdf_row)[part:nparts:end])
    findall(in(syss), pde_df.gdf_row)
end

function common_params()
    (;
        save_step=10,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=10 * 60 * 60,
        run_threads=10,
        solver_threads=nothing,
    )
end

function main4_part1()
    run_pde_setup("systems.jld2", "data";
        rows=part_rows("systems.jld2", 1, 3),
        common_params()...
    )
end

function main4_part2()
    run_pde_setup("systems.jld2", "data";
        rows=part_rows("systems.jld2", 2, 3),
        common_params()...
    )
end

function main4_part3()
    run_pde_setup("systems.jld2", "data";
        rows=part_rows("systems.jld2", 3, 3),
        common_params()...
    )
end

# The whole 60 row set at solver_threads=4 - no `rows`, so run_pde_setup takes
# all of pde_df - into its own data_st4_all, on a whole 96G node: 10 rows x 4
# threads is exactly its 40 cores. BLAS is pinned to one thread by the call at
# the top of this file, so those 4 are the only threading left in play, unlike
# the earlier data_st4 pass which ran before that fix.
function main4_all_st4()
    run_pde_setup("systems.jld2", "data_st4_all";
        merge(common_params(), (; solver_threads=4))...
    )
end
