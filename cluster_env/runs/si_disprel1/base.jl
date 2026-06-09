using Revise
using SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.MinimalModelV2, SSMCMain.ModifiedMiCRM.TwoMMs

include("../../../scripts/single_influx.jl")

using Base.Threads, OhMyThreads
using Printf
using ProgressMeter
using JLD2, Geppetto
using Random, Distributions
using DataFrames, DataFramesMeta
using Optim

function solve_si_odes(
    outfname, num_runs,
    K, l, p,
    T, tol;
    DN=0.,
    N=20, M=20,
    si_u0=[fill(1., N); fill(0., M)],
    solver=TRBDF2,
    maxtime=30.,
    surv_threshold=1e-9,
)
    rsg = get_si_sampler_for_paper(K, l, DN; DR=p, N, M)
    N = rsg.Ns
    metadata = (;
        num_runs, K, l, p, T, tol, DN, N, M, si_u0, solver, maxtime, surv_threshold,
        rsg,
    )

    params = Vector{BMMiCRMParams}(undef, num_runs)
    Dss = Vector{Vector{Float64}}(undef, num_runs)
    retcodes = Vector{ReturnCode.T}(undef, num_runs)
    final_states = Vector{Vector{Float64}}(undef, num_runs)
    maxresids = Vector{Float64}(undef, num_runs)
    num_surv = Vector{Int}(undef, num_runs)

    prog = Progress(num_runs)
    @tasks for i in 1:num_runs
        gen_ps = rsg()
        si_ps = gen_ps.mmicrm_params
        si_Ds = gen_ps.Ds

        si_p = make_mmicrm_problem(si_ps, si_u0, T)
        si_s = solve(si_p, solver();
            dense=false,
            save_everystep=false,
            callback=CallbackSet(make_timer_callback(maxtime), PositiveDomain(si_u0)),
            abstol=tol,
            reltol=tol,
        )
        si_fs = si_s.u[end]
        si_num_surv = count(>(surv_threshold), si_fs[1:N])

        params[i] = si_ps
        Dss[i] = si_Ds
        retcodes[i] = si_s.retcode
        final_states[i] = si_fs
        maxresids[i] = mmicrmmaxresid(si_s)

        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)

    df = DataFrame(; params, Dss, retcodes, final_states, maxresids, num_runs)
    jldsave(outfname; metadata, df)
    df
end

function main1()
    Klps_to_run = []
    for p in [1., 0.1]
        for l in [1., 0.999, 0.99, 0.9, 0.8]
            for K in range(10^0.5, 10^2, 10)
                push!(Klps_to_run, (K, l, p))
            end
        end
    end
    Klps_to_run
    for ri in 1:length(Klps_to_run)
        K, l, p = Klps_to_run[ri]
        @printf("Running %d/%d: K=%.3f, l=%.3f, p=%.3f\n", ri, length(Klps_to_run), K, l, p)
        flush(stdout)

        solve_si_odes("main1_$(ri).jld2", 100,
            K, l, p,
            1e8, 1e-9,
        )
    end
end
