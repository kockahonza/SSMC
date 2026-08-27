################################################################################
# For Fig.3e, multiple systems each solved at 5 different p
################################################################################
include("../base.jl")

function main1()
    run_pde_setup("systems.jld2", "data";
        save_step=nothing,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=5 * 60 * 60,
        run_threads=40,
        solver_threads=nothing,
    )
end

function solve_more_si_odes()
    K = 10.
    l = 0.999
    T = 1e8
    tol = 1e-9
    @time df = solve_si_odes("test2.jld2", 10000,
        K, l,
        T, tol,
    );
    @show count(df.good), nrow(df)
end
