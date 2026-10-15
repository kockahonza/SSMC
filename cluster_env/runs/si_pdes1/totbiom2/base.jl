include("../base.jl")

function main1()
    println("Starting prep")
    flush(stdout)
    make_pde_setup("../ode_solves.jld2", "systems1.jld2",
        20,                                                # num_systems
        10 .^ range(-2, 2, 1000),                          # ks, grid the disprel peaks are read off
        10 .^ range(-2, 0, 5),                             # ps, byproduct diffusion DR
        1e-3,                                              # DN, strain diffusion
        1.0,                                               # DI, influx resource diffusion
        30,                                                # num_peaks, L fits this many peaks of the peak k
        2000,                                              # sN, spatial gridpoints
    )

    println("Finished prep, starting PDE runs")
    flush(stdout)
    run_pde_setup("systems1.jld2", "data1";
        save_step=20,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=3 * 60 * 60,
        run_threads=40,
        solver_threads=nothing,
    )
end
