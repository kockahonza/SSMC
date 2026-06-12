using Revise
includet("../base.jl")


function change_p(ps, p; DI=1.)
    N, M = get_Ns(ps)
    iri = findfirst(!iszero, ps.K)

    new_ps = copy(ps);
    new_ps.Ds[N+1:N+M] .= p * DI
    new_ps.Ds[N+iri] = DI

    new_ps
end

function main1()
    T = 1e8
    L = 5
    sN = 1000
    sp_epsilon = 1e-3
    tol = 1e-9
    dx = L / sN

    ps = 10 .^ range(0, -2, 8)

    save_step = 500

    maxtime = 3 * 60 * 60
    run_threads=8
    solver_threads=16

    # Load ode final state
    f = jldopen("./sys1.jld2")
    gen_ps = f["gen_ps"]
    ode_s = f["ode_s"]
    ode_fs = f["ode_fs"]

    N, M = get_Ns(gen_ps)
    iri = findfirst(!iszero, gen_ps.K)

    # Generate the same u0 for all sims
    pde_u0 = expand_u0_to_size((sN,), ode_fs)
    pde_u0 = perturb_u0_uniform(N, M, pde_u0, sp_epsilon)
    clamp!(pde_u0, 0., Inf);

    # Run the sims
    params = Vector{Any}(undef, length(ps))
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
        pde_s = solve(pde_p, TRBDF2();
            dense=false,
            save_everystep=false,
            abstol=tol,
            reltol=tol,
            callback=CallbackSet(make_timer_callback(maxtime), PositiveDomain(copy(pde_u0); save=false), scb),
        );
        push!(pde_s_t, pde_s.t[end])
        push!(pde_s_u, pde_s.u[end])

        params[i] = pde_ps
        sol_ts[i] = pde_s_t
        sol_us[i] = pde_s_u
        maxresids[i] = smmicrmmaxresid(pde_s)
    end

    df = DataFrame(;
        ps, params, sol_ts, sol_us, maxresids
    )

    jldsave("./data1.jld2"; df)
end
