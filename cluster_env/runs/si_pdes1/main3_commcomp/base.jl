################################################################################
# For Fig.3e, multiple systems each solved at 5 different p
################################################################################
include("../base.jl")

function main1()
    run_pde_setup("systems.jld2", "data";
        save_step=nothing,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=10 * 60 * 60,
        run_threads=40,
        solver_threads=nothing,
    )
end

function main2()
    run_pde_setup("systems.jld2", "data2";
        save_step=nothing,
        solver=TRBDF2,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=10 * 60 * 60,
        run_threads=40,
        solver_threads=nothing,
    )
end
