using SSMCMain, SSMCMain.ModifiedMiCRM.MinimalModelV2
import SSMCMain.ModifiedMiCRM.MinimalModelV2

using Base.Threads
using OhMyThreads

using ProgressMeter
using JLD2

using Peaks

function do_DR_run(
    DRs, T,
    K, l,
    m, c,
    DN, DI,
    N0, epsilon,
    L, sN;
    outfname="v1_run_" * timestamp() * ".jld2",
    # solver stuff
    solver=QNDF,
    tol=10000 * eps(),
    maxtime=30 * 60,
    ns_maxtime=maxtime / 10,
    # threading
    run_threads=1,
    total_threads=nthreads(),
    solver_threads=div(total_threads, run_threads),
    kwargs...
)
    mmp = MMParams(;
        K, m,
        c, l,
        k=0.0,
        d=1.0,
    )
    ps = mmp_to_mmicrm(mmp)

    dx = L / sN

    params = Vector{Any}(undef, length(DRs))
    sp_retcodes = Vector{ReturnCode.T}(undef, length(DRs))
    sp_final_Ts = Vector{Float64}(undef, length(DRs))
    sp_final_states = Vector{Matrix{Float64}}(undef, length(DRs))
    sp_final_avgNs = Vector{Float64}(undef, length(DRs))

    @sprintf "Starting run for %s with %d DRs" outfname length(DRs)
    flush(stdout)

    prog = Progress(length(DRs))
    @tasks for ri in 1:length(DRs)
        @set ntasks = run_threads
        DR = DRs[ri]

        # Run spatial
        sps = SASMMiCRMParams(
            ps,
            SA[DN, DI, DR],
            make_cartesianspace_smart(1; dx),
            solver_threads
        )
        sp_u0 = fill(0.0, 3, sN)
        sp_u0[1, :] .= N0 .+ epsilon .* randn(sN)
        sp_prob = make_smmicrm_problem(sps, sp_u0, T)

        s = solve(sp_prob, solver();
            dense=false,
            save_everystep=false,
            abstol=tol,
            reltol=tol,
            callback=make_timer_callback(maxtime),
            kwargs...
        )

        params[ri] = sps # save the spatial params as they contain the diffusions too
        sp_retcodes[ri] = s.retcode
        sp_final_states[ri] = s.u[end]
        sp_final_avgNs[ri] = mean(s.u[end][1, :])
        sp_final_Ts[ri] = s.t[end]

        GC.gc()

        next!(prog)
        flush(stdout)
    end
    finish!(prog)

    # Run no-space
    @sprintf "Running no-space for %s" outfname
    flush(stdout)

    ns_prob = make_mmicrm_problem(ps, [N0, 0.0, 0.0], T)
    s = solve(ns_prob, solver();
        dense=false,
        save_everystep=false,
        abstol=tol,
        reltol=tol,
        callback=make_timer_callback(ns_maxtime),
        kwargs...
    )

    # Save data
    jldsave(outfname;
        DRs, T,
        K, l, m, c, DN, DI,
        N0, epsilon, L, sN,
        solver=string(solver), tol,
        params,
        sp_retcodes,
        sp_final_Ts,
        sp_final_states,
        sp_final_avgNs,
        ns_retcode=s.retcode,
        ns_final_T=s.t[end],
        ns_final_state=s.u[end],
        ns_final_avgN=s.u[end][1],
    )

    @sprintf "Finished run for %s" outfname
    flush(stdout)

    outfname
end

function get_peaks(fss, dx; minprom=0.1)
    full_peaks = map(fss) do fs
        all_pks = findmaxima(fs[1, :])
        peakproms(all_pks; min=minprom)
    end
    numpeaks = map(full_peaks) do pks
        length(pks.indices)
    end
    avg_pkh = map(full_peaks) do pks
        mean(pks.heights)
    end
    avg_pkw = map(full_peaks) do pks
        if length(pks.indices) > 0
            mean(peakwidths(pks).widths) * dx
        else
            missing
        end
    end
    avg_pkp = map(full_peaks) do pks
        if length(pks.indices) > 0
            mean(peakproms(pks).proms)
        else
            missing
        end
    end
    avg_pksp = map(full_peaks) do pks
        if length(pks.indices) > 1
            mean(pks.indices[2:end] .- pks.indices[1:end-1]) * dx
        else
            missing
        end
    end
    (; full_peaks, numpeaks, avg_pkh, avg_pkw, avg_pkp, avg_pksp)
end
