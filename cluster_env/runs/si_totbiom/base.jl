using Revise
using SSMCMain, SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.RandomSystems

include("../../../scripts/single_influx.jl")

using Base.Threads, OhMyThreads
using Printf
using ProgressMeter
using JLD2, Geppetto
using Random, Distributions


function change_p(ps, p; DI=1.)
    N, M = get_Ns(ps)
    iri = findfirst(!iszero, ps.K)

    new_ps = copy(ps);
    new_ps.Ds[N+1:N+M] .= p * DI
    new_ps.Ds[N+iri] = DI

    new_ps
end

function run_1_system_changing_p(
    outfname, gen_ps, ode_fs, ps,
    T, L, sN,
    sp_epsilon,
    save_step;
    tol=1e-9,
    maxtime=6 * 60 * 60,
    run_threads=8,
    solver_threads=div(nthreads(), run_threads),
    solver=TRBDF2,
)
    dx = L / sN

    N, M = get_Ns(gen_ps)
    iri = findfirst(!iszero, gen_ps.K)

    # Generate the same u0 for all sims
    pde_u0 = expand_u0_to_size((sN,), ode_fs)
    pde_u0 = perturb_u0_uniform(N, M, pde_u0, sp_epsilon)
    clamp!(pde_u0, 0., Inf);

    metadata = (;
        gen_ps, ode_fs, ps,
        T, L, sN,
        sp_epsilon,
        save_step,
        tol, maxtime,
        run_threads, solver_threads,
        dx, N, M, iri, pde_u0,
    )

    # Run the sims
    params = Vector{Any}(undef, length(ps))
    retcodes = Vector{ReturnCode.T}(undef, length(ps))
    sol_ts = Vector{Vector{Float64}}(undef, length(ps))
    sol_us = Vector{Vector{Matrix{Float64}}}(undef, length(ps))
    maxresids = Vector{Float64}(undef, length(ps))

    @localize pde_u0 @tasks for i in 1:length(ps)
        @set ntasks = run_threads
        @printf "Starting %d\n" i
        p = ps[i]

        local_ps = change_p(gen_ps, p)

        pde_ps = BSMMiCRMParams(
            local_ps.mmicrm_params,
            local_ps.Ds,
            CartesianSpace{1,Tuple{Periodic}}(SA[dx]),
            solver_threads
        )
        pde_p = make_smmicrm_problem(pde_ps, pde_u0, T; jac_type=:sparse);

        pde_s_t, pde_s_u, scb = make_stepped_saver_callback(pde_u0, save_step)
        push!(pde_s_t, 0.)
        push!(pde_s_u, pde_u0)
        pde_s = solve(pde_p, solver();
            dense=false,
            save_everystep=false,
            abstol=tol,
            reltol=tol,
            callback=CallbackSet(make_timer_callback(maxtime), PositiveDomain(copy(pde_u0); save=false), scb),
        );
        push!(pde_s_t, pde_s.t[end])
        push!(pde_s_u, pde_s.u[end])

        params[i] = pde_ps
        retcodes[i] = pde_s.retcode
        sol_ts[i] = pde_s_t
        sol_us[i] = pde_s_u
        maxresids[i] = smmicrmmaxresid(pde_s)
    end

    df = DataFrame(;
        ps, params, retcodes, sol_ts, sol_us, maxresids
    )

    jldsave(outfname; df, metadata)
end

