using Revise
includet("../base.jl")


function main1()
    ps = 10 .^ range(0, -2, 8)

    T = 1e8
    L = 5
    sN = 1000
    sp_epsilon = 1e-3

    save_step = 200
    run_threads = 8
    
    f = jldopen("./systems1.jld2")
    systems = f["systems"]
    fmd = f["metadata"]

    for (i, (gen_ps, ode_fs)) in enumerate(systems)
        outfname = 
        run_1_system_changing_p(
            "data1_better.jld2", gen_ps, ode_fs, ps,
            T, L, sN,
            sp_epsilon,
            save_step;
            tol=1e-9,
            maxtime=8 * 60 * 60,
            run_threads,
            solver_threads=div(nthreads(), run_threads),
        )
    end






    f = jldopen("./sys1.jld2")
    gen_ps = f["gen_ps"]
    ode_fs = f["ode_fs"]

    ps = 10 .^ range(0, -2, 8)

    T = 1e8
    L = 5
    sN = 1000
    sp_epsilon = 1e-3

    save_step = 200
    run_threads=8

    run_1_system_changing_p(
        "data1_better.jld2", gen_ps, ode_fs, ps,
        T, L, sN,
        sp_epsilon,
        save_step;
        tol=1e-9,
        maxtime=6 * 60 * 60,
        run_threads,
        solver_threads=div(nthreads(), run_threads),
    )
end
