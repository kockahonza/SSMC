using Revise
using SSMCMain, SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.RandomSystems

using Base.Threads, OhMyThreads
using Printf
using ProgressMeter
using JLD2, Geppetto
using Random, Distributions

################################################################################
# Running PDEs
################################################################################
function run_1d_pde_sim(ps, u0, T, L, sN;
    maxtime=60,
    solver_threads=nothing,
    tol=100000 * eps(),
    solver=QNDF,
)
    dx = L / sN

    sps = BSMMiCRMParams(
        ps.mmicrm_params,
        ps.Ds,
        CartesianSpace{1,Tuple{Periodic}}(SA[dx]),
        solver_threads
    )
    sp = make_smmicrm_problem(sps, u0, T)

    s = solve(sp, solver();
        dense=false,
        save_everystep=false,
        calck=false,
        abstol=tol,
        reltol=tol,
        callback=make_timer_callback(maxtime)
    )

    s
end

""" Fairly general function to run PDEs for all the systems in a given file """
function do_pde_runs(systems_fname, out_fname, T, L, sN, sp_epsilon, pde_solve_maxtime, run_threads;
    solver_threads = div(nthreads(), run_threads)
)
    metadata = (;
        T, L, sN, sp_epsilon,
        pde_solve_maxtime, run_threads, solver_threads,
        systems_fname, out_fname
    )

    f = jldopen(systems_fname, "r")
    systems_to_run = f["systems_to_run"]
    close(f)

    pde_retcodes = Vector{Vector{ReturnCode.T}}(undef, length(systems_to_run))
    pde_final_states = Vector{Vector{Matrix{Float64}}}(undef, length(systems_to_run))
    pde_final_Ts = Vector{Vector{Float64}}(undef, length(systems_to_run))

    total_runs = sum(systems_to_run) do x length(x.systems) end

    prog1 = Progress(total_runs)
    for groupi in 1:length(systems_to_run)
        systems = systems_to_run[groupi].systems
        num_systems = length(systems)
        push!(pde_retcodes, Vector{ReturnCode.T}(undef, num_systems))
        push!(pde_final_states, Vector{Matrix{Float64}}(undef, num_systems))
        push!(pde_final_Ts, Vector{Float64}(undef, num_systems))

        @tasks for si in 1:num_systems
            @set ntasks = run_threads

            ps = systems[si].params
            odess = systems[si].odess
            u0_ = expand_u0_to_size((sN,), odess)
            u0 = perturb_u0_uniform(get_Ns(ps)..., u0_, sp_epsilon)

            s = run_1d_pde_sim(ps, u0, T, L, sN;
                maxtime=pde_solve_maxtime,
                solver_threads,
            )

            pde_retcodes[end][si] = s.retcode
            pde_final_states[end][si] = s.u[end]
            pde_final_Ts[end][si] = s.t[end]

            s = nothing
            GC.gc()

            next!(prog1)
            flush(stdout)
        end
    end
    finish!(prog1)
    flush(stdout)

    jldsave(out_fname;
        systems_to_run,
        pde_retcodes, pde_final_states, pde_final_Ts,
        metadata,
    )

    (; pde_retcodes, pde_final_states, pde_final_Ts)
end


################################################################################
# Main functions
################################################################################
"""
Running PDEs for systems selected in systems1.jld2 -- running only a relatively small subset of Ks and lis and only 20 random runs at each
"""
function main1()
    do_pde_runs(
        "systems1.jld2", "data1.jld2",
        1000000000, 5, 5000, 1e-3, # T, L, sN, sp_epsilon
        5 * 60 * 60, # max time per pde run
        8, # number of simultaneous runs
    )
end

"""
systems2.jld2 -- same as main1 but much fewer lis
"""
function main2()
    do_pde_runs(
        "systems2.jld2", "data2.jld2",
        1000000000, 5, 5000, 1e-3, # T, L, sN, sp_epsilon
        5 * 60 * 60, # max time per pde run
        8, # number of simultaneous runs
    )
end

"""
systems3.jld2 -- same as main2 but just li=1 and a lower max solver time
"""
function main3()
    do_pde_runs(
        "systems3_just_l1.jld2", "data3.jld2",
        1000000000, 5, 5000, 1e-3, # T, L, sN, sp_epsilon
        2 * 60 * 60, # max time per pde run
        8, # number of simultaneous runs
    )
end
