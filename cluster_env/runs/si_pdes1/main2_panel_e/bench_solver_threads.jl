################################################################################
# Is solver_threads worth it for main2/main3?
################################################################################
# main2/main3 run 10 rows at a time on a 40 core node, so there are 30 cores
# spare unless each solver is given some. run_pde_setup's docstring says to hand
# the leftovers out when there are fewer rows than cores, but whether QNDF here
# actually uses them is a question about these systems - most of the time goes
# into the sparse linear solve, which solver_threads does not touch.
#
# Run under the same shape as the real job (10 rows at once, sN=2500), capping
# each row with maxtime and comparing how far through the integration it got.
# Same rows and same tolerances throughout, only the thread count changes, so
# more simulated time in the same wall clock means faster. save_step=nothing
# keeps it from writing the state series - this is a timing test, not a run.
include("base.jl")

function bench_solver_threads(setup_fname="systems1.jld2";
    nrows=10,
    maxtime=5 * 60,
    solver_threadss=(nothing, 2, 4),
    abstol=1e-7,
    reltol=1e-9,
)
    pde_df = load(setup_fname)["pde_df"]
    rows = collect(1:min(nrows, nrow(pde_df)))
    @printf("%d rows at a time, %ds cap each, sN=%d\n", length(rows), maxtime, pde_df.sN[1])
    flush(stdout)

    results = []
    for st in solver_threadss
        reached = zeros(length(rows))
        retcodes = Vector{Any}(undef, length(rows))
        wall = @elapsed @tasks for j in eachindex(rows)
            @set ntasks = length(rows)
            r = pde_df[rows[j], :]
            (; s) = run_si_pde(r.params, r.Ds, r.u0, r.T, r.L;
                abstol, reltol, maxtime, solver_threads=st, save_step=nothing,
            )
            reached[j] = s.t[end]
            retcodes[j] = s.retcode
        end
        GC.gc()

        push!(results, (; solver_threads=st, wall, reached=copy(reached), retcodes=copy(retcodes)))
        @printf("solver_threads=%-4s wall %6.1fs  total t %.4g  per row %s\n",
            isnothing(st) ? "1" : string(st), wall, sum(reached),
            string(round.(reached; sigdigits=3)))
        # anything but Terminated means the row stopped for its own reasons and
        # its simulated time is not comparable with the rest
        allequal(retcodes) || println("    retcodes: ", join(unique(retcodes), ", "))
        flush(stdout)
    end

    base = results[1]
    println("--- speedup vs solver_threads=1 (total simulated t, wall-normalised) ---")
    for r in results
        @printf("  solver_threads=%-4s %.2fx\n",
            isnothing(r.solver_threads) ? "1" : string(r.solver_threads),
            (sum(r.reached) / r.wall) / (sum(base.reached) / base.wall))
    end
    flush(stdout)

    results
end
