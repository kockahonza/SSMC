include("../base.jl")
using LinearAlgebra

function main1()
    run_pde_setup("systems.jld2", "data";
        save_step=20,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=3 * 60 * 60,
        run_threads=40,
        solver_threads=nothing,
    )
end
