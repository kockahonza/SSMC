include("../../../scripts/single_influx.jl")

BLAS.set_num_threads(1)

using Distributions

function beta_from_mean_std(mu, sigma)
    @assert 0 < mu < 1 "mean must be in (0, 1), got $mu"
    @assert sigma > 0 "std must be positive, got $sigma"
    @assert sigma^2 < mu * (1 - mu) "std too large for this mean, need sigma < sqrt(mu*(1-mu)) = $(sqrt(mu * (1 - mu)))"
    nu = mu * (1 - mu) / sigma^2 - 1  # alpha + beta
    Beta(mu * nu, (1 - mu) * nu)
end
function beta_from_mean_sigmaf(mu, sigmaf)
    ldist = beta_from_mean_std(mu, max_sigma_not_bimodal(mu) * sigmaf)
end

max_allowable_sigma(mu) = sqrt(mu * (1 - mu))
function max_sigma_for_unimodal(mu)
    m = min(mu, 1 - mu)
    sqrt(mu * (1 - mu) * (m / (1 + m)))
end
function max_sigma_not_bimodal(mu)
    m = max(mu, 1 - mu)
    sqrt(mu * (1 - mu) * (m / (1 + m)))
end

function do_run_Klrand2(Ks, lmeans, lsigmafs, num_repeats;
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
    numKs, numlmeans, numlsigmafs = length(Ks), length(lmeans), length(lsigmafs)
    num_runs = numKs * numlmeans * numlsigmafs * num_repeats

    N = haskey(rsg_kwargs, :N) ? rsg_kwargs[:N] : 20
    M = haskey(rsg_kwargs, :M) ? rsg_kwargs[:M] : N
    u0=[fill(1., N); fill(0., M)]

    metadata = (;
        Ks, lmeans, lsigmafs, num_repeats,
        DN, rsg_kwargs,
        N, M, u0,
        T, solver, tol, abstol, reltol, maxtime, use_extinction_callback, extinction_threshold, maxresid_threshold, save_all_ps,
        lsks
    )

    # keeping track of indices to be safe
    Kis = Vector{Int}(undef, num_runs)
    lmeanis = Vector{Int}(undef, num_runs)
    lsigmafis = Vector{Int}(undef, num_runs)

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

    rsgs = Array{JansSampler3}(undef, numKs, numlmeans, numlsigmafs)
    for Ki in 1:numKs, lmeani in 1:numlmeans, lsigmafi in 1:numlsigmafs
        ldist = beta_from_mean_sigmaf(lmeans[lmeani], lsigmafs[lsigmafi])
        rsgs[Ki, lmeani, lsigmafi] = get_si_sampler_for_paper(Ks[Ki], ldist, DN; rsg_kwargs...)
    end

    cis = CartesianIndices((numKs, numlmeans, numlsigmafs, num_repeats))
    pb = Progress(num_runs)
    @tasks for row_i in 1:num_runs
        @set scheduler = :greedy

        Ki, lmeani, lsigmafi, _ = Tuple(cis[row_i])
        rsg = rsgs[Ki, lmeani, lsigmafi]

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
        lmeanis[row_i] = lmeani
        lsigmafis[row_i] = lsigmafi
        params[row_i] = save_ps ? ps : nothing
        codes[row_i] = code

        next!(pb)
        flush(stdout)
    end
    finish!(pb)
    flush(stdout)

    df = DataFrame(;
        Kis,
        lmeanis,
        lsigmafis,
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
    lmeanis = 1:maximum(df.lmeanis)
    numlmeans = length(lmeanis)
    lsigmafis = 1:maximum(df.lsigmafis)
    numlsigmafs = length(lsigmafis)

    codes = sort(unique(df.codes))
    matrices = Dict{Int,Array{Int,3}}()
    for code in codes
        mat = matrices[code] = zeros(Int, numKs, numlmeans, numlsigmafs)
        for r in eachrow(df)
            if r.codes == code
                mat[r.Kis, r.lmeanis, r.lsigmafis] += 1
            end
        end
    end
    matrices
end

function main1()
    Ks = 10 .^ range(0.2, 3.1, 15)
    lmeans = [0.75, 0.9, 0.99, 0.999]
    lsigmafactors = range(3e-2, 1/sqrt(3), 4) .^ 2

    df, metadata = do_run_Klrand2(Ks, lmeans, lsigmafactors, 150;
        T=1e6,
        maxtime=120,
        rsg_kwargs=(;)
    )
    jldsave("./main1.jld2"; df, metadata)
end

function main2()
    Ks = 10 .^ range(0., 3.5, 30)
    leak_xs = range(0.0, LeakageScale.ltox(0.999), 15)
    lmeans = LeakageScale.l.(leak_xs)
    lsigmafactors = range(3e-2, 1/sqrt(3), 4) .^ 2

    df, metadata = do_run_Klrand2(Ks, lmeans, lsigmafactors, 150;
        T=1e6,
        maxtime=120,
        rsg_kwargs=(;)
    )
    jldsave("./main2.jld2"; df, metadata)
end
