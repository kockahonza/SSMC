include("../../../scripts/single_influx.jl")

BLAS.set_num_threads(1)

function do_Kli_run(Ks, lis, num_repeats;
    DN=0.,
    rsg_kwargs=(;),
    T=1e8,
    # ODE solver params
    solver=TRBDF2,
    tol=1e-9,
    abstol=100 * tol,
    reltol=tol,
    maxtime=30.,
    extinction_threshold=10 * abstol,
    maxresid_threshold=10 * abstol,
    save_all_ps=false,
    # linear stability ks/qs to test at
    ls_threshold=tol,
    lsks=10 .^ range(-5, 3, 2000),
)
    num_runs = length(Ks) * length(lis) * num_repeats

    N = haskey(rsg_kwargs, :N) ? rsg_kwargs[:N] : 20
    M = haskey(rsg_kwargs, :M) ? rsg_kwargs[:M] : N
    u0=[fill(1., N); fill(0., M)]

    metadata = (;
        Ks, lis, num_repeats,
        DN, rsg_kwargs,
        N, M, u0,
        T, solver, tol, abstol, reltol, maxtime, extinction_threshold, maxresid_threshold, save_all_ps,
        lsks
    )

    # keeping track of indices to be safe
    Kis = Vector{Int}(undef, num_runs)
    liis = Vector{Int}(undef, num_runs)

    params = Vector{Union{Nothing,BSMMiCRMParams}}(undef, num_runs)
    # solver debugging
    retcodes = Vector{ReturnCode.T}(undef, num_runs)
    final_Ts = Vector{Float64}(undef, num_runs)
    maxresids = Vector{Float64}(undef, num_runs)
    num_iters = Vector{Int}(undef, num_runs)
    # well-mixed steady state
    final_states = Vector{Vector{Float64}}(undef, num_runs)
    # outcome code
    codes = Vector{Int}(undef, num_runs)

    row_i_ = 1
    prog = Progress(length(Ks) * length(lis))
    for Ki in 1:length(Ks)
        for lii in 1:length(lis)
            K = Ks[Ki]
            li = lis[lii]
            rsg = get_si_sampler_for_paper(K, li, DN; rsg_kwargs...)

            rows = row_i_:(row_i_+num_repeats-1)
            @tasks for row_i in rows
                @set scheduler = :greedy
                ps = rsg()
                save_ps = save_all_ps

                prob = make_mmicrm_problem(ps.mmicrm_params, copy(u0), T)
                sol = solve(prob, solver();
                    dense=false,
                    save_everystep=false,
                    callback=CallbackSet(
                        make_timer_callback(maxtime),
                        make_ode_extinction_exit_callback(N, extinction_threshold),
                        PositiveDomain(copy(u0); save=false),
                    ),
                    abstol=abstol,
                    reltol=reltol,
                )
                ss = sol.u[end]

                retcodes[row_i] = sol.retcode
                final_Ts[row_i] = sol.t[end]
                maxresids[row_i] = mmicrmmaxresid(sol)
                num_iters[row_i] = sol.stats.naccept
                final_states[row_i] = ss

                code = if sol.retcode != ReturnCode.Success
                    save_ps = true
                    -1
                elseif maxresids[row_i] > maxresid_threshold
                    save_ps = true
                    -3
                elseif maximum(ss[1:N]) < extinction_threshold
                    1
                else # do linear stability analysis
                    M1 = make_M1(ps, ss)
                    if maximum(real, eigvals(M1)) > -ls_threshold # the system is ecologically unstable in the well-mixed case
                        -2
                    else
                        ls_code = 2
                        for k in lsks
                            M_ = M1_to_M(M1, ps.Ds, k)
                            if maximum(real, eigvals(M_)) > ls_threshold # found a spatial instability
                                ls_code = 3
                                break
                            end
                        end
                        ls_code
                    end
                end

                Kis[row_i] = Ki
                liis[row_i] = lii
                params[row_i] = save_ps ? ps : nothing
                codes[row_i] = code
            end

            row_i_ += num_repeats
            next!(prog)
            flush(stdout)
        end
    end
    finish!(prog)
    flush(stdout)

    df = DataFrame(;
        Kis,
        liis,
        params,
        retcodes,
        final_Ts,
        maxresids,
        num_iters,
        final_states,
        codes,
    )

    df, metadata
end

function main1()
    Ks = 10 .^ range(-0.5, 4.0, 50)
    leak_xs = range(0.0, LeakageScale.ltox(0.999), 30)
    lis = LeakageScale.l.(leak_xs)

    df, metadata = do_Kli_run(Ks, lis, 110;
        T=1e6,
        maxtime=120,
    )
    jldsave("./main1.jld2"; df, metadata)
end
