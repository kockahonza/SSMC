################################################################################
# For Fig.3e, multiple systems each solved at 5 different p
################################################################################
include("../base.jl")

function main1()
    run_pde_setup("systems1_smallL.jld2", "data1_smallL";
        save_step=20,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=1 * 60 * 60,
        run_threads=40,
        solver_threads=nothing,
    )
end

function main2()
    run_pde_setup("systems2_bigL.jld2", "data2_bigL";
        save_step=20,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=1 * 60 * 60,
        run_threads=40,
        solver_threads=nothing,
    )
end
