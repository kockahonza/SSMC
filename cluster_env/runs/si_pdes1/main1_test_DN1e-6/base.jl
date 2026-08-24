include("../base.jl")

"""
First PDE run off the notebook-made setup file. 9 rows (3 systems x 3 p), all
9 at once with 3 threads each. 27 cores rather than a whole node because BIOP
nodes are only ever partly free - see job1.slurm.
"""
function main1()
    run_pde_setup("pde_setup1.jld2", "main1";
        save_step=100,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=10 * 60 * 60,
        run_threads=9,
        solver_threads=nothing,
    )
end
