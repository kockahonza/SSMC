using Revise
includet("../base.jl")


function main1()
    ps = 10 .^ range(0, -2, 5)

    T = 1e8
    L = 5
    sN = 1000
    sp_epsilon = 1e-3

    save_step = 200
    run_threads = 8

    f = jldopen("./systems1_16.jld2")
    systems = f["systems"]

    println("Starting")
    flush(stdout)
    prog = Progress(length(systems))
    @tasks for i in 1:length(systems)
        @set ntasks = run_threads

        gen_ps, ode_fs = systems[i]
        outfname = "./data1/r$i.jld2"
        run_1_system_changing_p(
            outfname, gen_ps, ode_fs, ps,
            T, L, sN,
            sp_epsilon,
            save_step;
            tol=1e-9,
            maxtime=8 * 60 * 60,
            run_threads=1,
            solver_threads=div(nthreads(), run_threads),
            solver=TRBDF2,
        )

        GC.gc()
        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)
end
