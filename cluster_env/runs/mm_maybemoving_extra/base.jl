using SSMCMain, SSMCMain.ModifiedMiCRM, SSMCMain.SpaceMMiCRM

using JLD2
using Base.Threads

function resample_sps(sps, N, extra_time;
    prop_perturb_param=nothing,
    usenthreads=nthreads()
)
    # resample the final u of the sps
    u0, new_space = resample_cartesian_u(
        sps.u[end],
        sps.prob.p.space,
        N
    )
    # make sure to update dx accordingly in the params, and update num threads
    new_smmicrm_params = change_smmicrm_params(sps.prob.p;
        space=new_space, usenthreads
    )

    # optionally also perturb the starting u
    if !isnothing(prop_perturb_param)
        u0 = perturb_u0_uniform_prop(1, 2, u0, prop_perturb_param)
    end

    # make the ODEProblem
    make_smmicrm_problem_safe(u0, extra_time, new_smmicrm_params; t0=sps.t[end])
end

function job1()
    mm_mm_sps = load_object("linput_mm_mm.jld2")

    extra_time = 10.0 * (mm_mm_sps.t[end] - mm_mm_sps.t[begin])

    nop = resample_sps(mm_mm_sps, 1000, extra_time)
    @time nop_sol = solve(nop, QNDF(); maxiters=10000)
    print_spatial_solution_stats(nop_sol)
    flush(stdout)
    @info "finished no perturbation run, saving"
    flush(stdout)
    save_object("nop_sol.jld2", nop_sol)

    nop = resample_sps(mm_mm_sps, 1000, extra_time;
        prop_perturb_param=0.05
    )
    flush(stdout)
    @time p5_sol = solve(nop, QNDF(); maxiters=10000)
    print_spatial_solution_stats(p5_sol)
    flush(stdout)
    @info "finished perturbation 0.05 run, saving"
    flush(stdout)
    save_object("p5_sol.jld2", p5_sol)
end
