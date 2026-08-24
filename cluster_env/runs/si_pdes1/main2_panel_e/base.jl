################################################################################
# For Fig.3e, multiple systems each solved at 5 different p
################################################################################
include("../base.jl")

function main1()
    run_pde_setup("systems.jld2", "data";
        save_step=100,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=10 * 60 * 60,
        run_threads=18,
        solver_threads=nothing,
    )
end
