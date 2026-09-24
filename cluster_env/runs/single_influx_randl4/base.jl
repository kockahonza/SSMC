include("../../../scripts/single_influx.jl")

BLAS.set_num_threads(1)

using Distributions

function do_run_Klrand4(Ks, lminmaxs, num_repeats;
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
    @assert all(lmm -> lmm[1] <= lmm[2], lminmaxs) "each lminmax must have min <= max"

    numKs, numlminmaxs = length(Ks), length(lminmaxs)
    num_runs = numKs * numlminmaxs * num_repeats

    N = haskey(rsg_kwargs, :N) ? rsg_kwargs[:N] : 20
    M = haskey(rsg_kwargs, :M) ? rsg_kwargs[:M] : N
    u0=[fill(1., N); fill(0., M)]

    metadata = (;
        Ks, lminmaxs, num_repeats,
        DN, rsg_kwargs,
        N, M, u0,
        T, solver, tol, abstol, reltol, maxtime, use_extinction_callback, extinction_threshold, maxresid_threshold, save_all_ps,
        lsks
    )

    # keeping track of indices to be safe
    Kis = Vector{Int}(undef, num_runs)
    lminmaxis = Vector{Int}(undef, num_runs)

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

    rsgs = Array{JansSampler3}(undef, numKs, numlminmaxs)
    for Ki in 1:numKs, lminmaxi in 1:numlminmaxs
        lmin, lmax = lminmaxs[lminmaxi]
        li = lmin == lmax ? Dirac(lmin) : Uniform(lmin, lmax)
        rsgs[Ki, lminmaxi] = get_si_sampler_for_paper(Ks[Ki], li, DN; rsg_kwargs...)
    end

    cis = CartesianIndices((numKs, numlminmaxs, num_repeats))
    pb = Progress(num_runs)
    @tasks for row_i in 1:num_runs
        @set scheduler = :greedy

        Ki, lminmaxi, _ = Tuple(cis[row_i])
        rsg = rsgs[Ki, lminmaxi]

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
        lminmaxis[row_i] = lminmaxi
        params[row_i] = save_ps ? ps : nothing
        codes[row_i] = code

        next!(pb)
        flush(stdout)
    end
    finish!(pb)
    flush(stdout)

    df = DataFrame(;
        Kis,
        lminmaxis,
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
    lminmaxis = 1:maximum(df.lminmaxis)
    numlminmaxs = length(lminmaxis)

    codes = sort(unique(df.codes))
    matrices = Dict{Int,Array{Int,2}}()
    for code in codes
        mat = matrices[code] = zeros(Int, numKs, numlminmaxs)
        for r in eachrow(df)
            if r.codes == code
                mat[r.Kis, r.lminmaxis] += 1
            end
        end
    end
    matrices
end

function main1()
    Ks = 10 .^ range(0., 3.5, 50)

    leak_xs = range(0.0, LeakageScale.ltox(0.999), 20)
    lis = LeakageScale.l.(leak_xs)
    lminmaxs = Tuple{Float64,Float64}[]
    for i1 in 1:length(lis)      # max
        for i2 in 1:length(lis)  # min
            if lis[i2] > lis[i1]
                break
            end
            push!(lminmaxs, (lis[i2], lis[i1]))
        end
    end
    @show lminmaxs

    df, metadata = do_run_Klrand4(Ks, lminmaxs, 150;
        T=1e6,
        maxtime=120,
        rsg_kwargs=(;)
    )
    jldsave("./main1.jld2"; df, metadata)
end
