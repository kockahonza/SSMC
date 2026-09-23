include("../../../scripts/single_influx.jl")

BLAS.set_num_threads(1)

using Distributions

function do_run_Klrand3(Ks, lalphas, lbetas, num_repeats;
    DN=0.,
    rsg_kwargs=(;),
    T=1e8,
    # ODE solver params
    solver=TRBDF2,
    tol=1e-9,
    abstol=100 * tol,
    reltol=tol,
    maxtime=30.,
    use_extinction_callback=false,
    extinction_threshold=10 * abstol,
    maxresid_threshold=10 * abstol,
    save_all_ps=false,
    # linear stability ks/qs to test at
    ls_threshold=tol,
    lsks=10 .^ range(-5, 3, 2000),
)
    numKs, numlalphas, numlbetas = length(Ks), length(lalphas), length(lbetas)
    num_runs = numKs * numlalphas * numlbetas * num_repeats

    N = haskey(rsg_kwargs, :N) ? rsg_kwargs[:N] : 20
    M = haskey(rsg_kwargs, :M) ? rsg_kwargs[:M] : N
    u0=[fill(1., N); fill(0., M)]

    metadata = (;
        Ks, lalphas, lbetas, num_repeats,
        DN, rsg_kwargs,
        N, M, u0,
        T, solver, tol, abstol, reltol, maxtime, use_extinction_callback, extinction_threshold, maxresid_threshold, save_all_ps,
        lsks
    )

    # keeping track of indices to be safe
    Kis = Vector{Int}(undef, num_runs)
    lalphais = Vector{Int}(undef, num_runs)
    lbetais = Vector{Int}(undef, num_runs)

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

    rsgs = Array{JansSampler3}(undef, numKs, numlalphas, numlbetas)
    for Ki in 1:numKs, lalphai in 1:numlalphas, lbetai in 1:numlbetas
        rsgs[Ki, lalphai, lbetai] = get_si_sampler_for_paper(Ks[Ki], Beta(lalphas[lalphai], lbetas[lbetai]), DN; rsg_kwargs...)
    end

    cis = CartesianIndices((numKs, numlalphas, numlbetas, num_repeats))
    pb = Progress(num_runs)
    @tasks for row_i in 1:num_runs
        @set scheduler = :greedy

        Ki, lalphai, lbetai, _ = Tuple(cis[row_i])
        rsg = rsgs[Ki, lalphai, lbetai]

        ps = rsg()
        save_ps = save_all_ps

        prob = make_mmicrm_problem(ps.mmicrm_params, copy(u0), T)
        sol = solve(prob, solver();
            dense=false,
            save_everystep=false,
            callback=CallbackSet(
                make_timer_callback(maxtime),
                PositiveDomain(copy(u0); save=false),
                use_extinction_callback ? make_ode_extinction_exit_callback(N, extinction_threshold) : nothing,
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
        lalphais[row_i] = lalphai
        lbetais[row_i] = lbetai
        params[row_i] = save_ps ? ps : nothing
        codes[row_i] = code

        next!(pb)
        flush(stdout)
    end
    finish!(pb)
    flush(stdout)

    df = DataFrame(;
        Kis,
        lalphais,
        lbetais,
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

function make_count_arrays(df)
    Kis = 1:maximum(df.Kis)
    numKs = length(Kis)
    lalphais = 1:maximum(df.lalphais)
    numlalphas = length(lalphais)
    lbetais = 1:maximum(df.lbetais)
    numlbetas = length(lbetais)

    codes = sort(unique(df.codes))
    matrices = Dict{Int,Array{Int,3}}()
    for code in codes
        mat = matrices[code] = zeros(Int, numKs, numlalphas, numlbetas)
        for r in eachrow(df)
            if r.codes == code
                mat[r.Kis, r.lalphais, r.lbetais] += 1
            end
        end
    end
    matrices
end

function main1()
    Ks = 10 .^ range(0., 3.5, 50)
    lalphas = 2 .^ range(-1, 6, 6)
    lbetas = copy(lalphas)

    df, metadata = do_run_Klrand3(Ks, lalphas, lbetas, 150;
        T=1e6,
        maxtime=120,
        rsg_kwargs=(;)
    )
    jldsave("./main1.jld2"; df, metadata)
end
