using SSMCMain, SSMCMain.ModifiedMiCRM.MinimalModelV2

using Base.Threads
using OhMyThreads

using JLD2

function run_explike_Kl_nospace(logKs, ls, T;
    m=1.0,
    k=0.0,
    c=1.0,
    d=1.0,
    u0=[1.0, 0.0, 0.0],
    save_sols=false,
    solver=TRBDF2(),
    kwargs...
)
    sols = Matrix{Any}(undef, length(logKs), length(ls))
    retcodes = Matrix{ReturnCode.T}(undef, length(logKs), length(ls))
    maxresids = Matrix{Float64}(undef, length(logKs), length(ls))
    final_states = Matrix{Vector{Float64}}(undef, length(logKs), length(ls))
    @tasks for i in 1:length(logKs)
        logK = logKs[i]
        for (j, l) in enumerate(ls)
            mmp = MMParams(;
                K=10^logK,
                m,
                l=l,
                k,
                c,
                d,
            )
            mmicrm_params = mmp_to_mmicrm(mmp)
            p = make_mmicrm_problem(mmicrm_params, copy(u0), T)
            s = solve(p, solver; kwargs...)
            # p = make_mmicrm_ss_problem(mmicrm_params, copy(u0))
            # s = solve(p, DynamicSS(solver); kwargs...)

            if save_sols
                sols[i, j] = s
            end
            retcodes[i, j] = s.retcode
            maxresids[i, j] = maximum(abs, uninplace(mmicrmfunc!)(s.u[end], mmicrm_params))
            final_states[i, j] = s.u[end]
        end
        @sprintf "Finished %d out of %d logK runs\n" i length(logKs)
    end
    (; retcodes, maxresids, final_states, sols=(save_sols ? sols : nothing))
end

function run_explike_Kl_space(logKs, ls, T;
    # unscanned of MM params
    m=1.0,
    k=0.0,
    c=1.0,
    d=1.0,
    # diffusions
    DN=1e-8, DI=1.0, DR=1e-8,
    # spatial domain
    L=2,
    gridsize=10000,
    # initial state
    u0=[1.0, 0.0, 0.0],
    epsilon=1e-5,
    # misc
    save_sols=false,
    solver=TRBDF2(),
    kwargs...
)
    diffs = SA[DN, DI, DR]
    dx = L / (gridsize + 1)

    space = make_cartesianspace_smart(1; dx)

    # use the same initial state for all runs
    u0 = reduce(hcat, [u0 .+ epsilon .* randn(3) for _ in 1:gridsize])

    sols = Matrix{Any}(undef, length(logKs), length(ls))
    retcodes = Matrix{ReturnCode.T}(undef, length(logKs), length(ls))
    final_abundances = Matrix{Vector{Float64}}(undef, length(logKs), length(ls))

    let u0 = u0
        @tasks for i in 1:length(logKs)
            logK = logKs[i]
            for (j, l) in enumerate(ls)
                mmp = MMParams(;
                    K=10^logK,
                    m,
                    l=l,
                    k,
                    c,
                    d,
                )
                p = SASMMiCRMParams(mmp_to_mmicrm(mmp), diffs, space, nothing)
                sp = make_smmicrm_problem(p, copy(u0), T)
                s = solve(sp, solver; kwargs...)

                if save_sols
                    sols[i, j] = s
                end
                retcodes[i, j] = s.retcode
                # sum(s.u[end]; dims=2)
                final_abundances[i, j] = sum(s.u[end]; dims=2)[:, 1]
            end
        end
    end
    (; retcodes, final_abundances, sols=(save_sols ? sols : nothing))
end

function main1()
    logKs = range(-0.5, 3, 100)
    lis = range(0.0, 1.0, 50)
    @time xx = run_explike_Kl_nospace(logKs, lis, 10000000;
        save_sols=true
    )

    @time jldsave("faf.jld2"; logKs, lis, xx)
end
