using SSMCMain, SSMCMain.ModifiedMiCRM.MinimalModelSemisymbolic

using DataFrames
using JLD2
using ChunkSplitters
using Base.Iterators
using Base.Threads

function analyze_many_mmps2(save_filename=nothing;
    m=[1.0],
    l=[1.0],
    K=[1.0],
    c=[1.0],
    d=[1.0],
    DN=[0.0],
    DG=[0.0],
    DR=[0.0],
    threshold=2 * eps(Float64),
    usenthreads=nothing
)
    params_nospace = collect(product(m, l, K, c, d))
    params_ci_nospace = CartesianIndices(params_nospace)
    params_diff = collect(product(DN, DG, DR))
    params_ci_diff = CartesianIndices(params_diff)

    results_array = zeros(Int, size(params_ci_nospace)..., size(params_ci_diff)...)

    @info "Starting the run"
    flush(stdout)
    if isnothing(usenthreads)
        for pnsci in params_ci_nospace
            analyze_single_mmps2_step_!(
                results_array,
                pnsci, params_ci_diff,
                params_nospace, params_diff,
                threshold
            )
        end
    else
        pns_chunks = chunks(params_ci_nospace; n=usenthreads, split=RoundRobin())

        @sync for (ti, ch) in enumerate(pns_chunks)
            @spawn begin
                @info (@sprintf "job %d starting on thread %d" ti threadid())
                flush(stdout)

                for pnsci in ch
                    analyze_single_mmps2_step_!(
                        results_array,
                        pnsci, params_ci_diff,
                        params_nospace, params_diff,
                        threshold
                    )
                end
            end
        end
    end

    if !isnothing(save_filename)
        jldsave(save_filename;
            data=results_array,
            m, l, K, c, d, DN, DG, DR,
            threshold
        )
    end

    results_array
end
function analyze_single_mmps2_step_!(
    results_array,
    pnsci, # the cartesian index of the params_nospace
    params_ci_diff, # the diff constant parameter index matrix
    params_nospace, params_diff, # the actual parameter arrays
    threshold
)
    lm, ll, lK, lc, ld = params_nospace[pnsci]
    mmpns = MinimalModelParamsNoSpace(lm, ll, lK, lc, ld)
    mmicrm_params = mmp_to_mmicrm(mmpns)

    ns_status, maybe_nssols = mmp_nospace_sol(mmpns)

    found_valid_ss = false
    debug_code = nothing

    nospace_stable_ss_M1 = nothing
    if ns_status == MMNoSpaceSolType.onepositive
        M1 = make_M1(mmicrm_params, maybe_nssols)

        M1evals = eigvals(M1)
        if all(l -> real(l) < -threshold, M1evals) # it is stable
            found_valid_ss = true
            nospace_stable_ss_M1 = M1
        else
            debug_code = -1
        end
    elseif ns_status == MMNoSpaceSolType.bothpositive
        M1p = make_M1(mmicrm_params, maybe_nssols[1])
        M1m = make_M1(mmicrm_params, maybe_nssols[2])

        pstable = all(l -> real(l) < -threshold, eigvals(M1p))
        mstable = all(l -> real(l) < -threshold, eigvals(M1m))

        if pstable && !mstable # we think this should always be the case
            found_valid_ss = true
            nospace_stable_ss_M1 = M1p
        elseif pstable && mstable
            debug_code = -1
        elseif !pstable && !mstable
            debug_code = -2
        else
            debug_code = -3
        end
    elseif isnothing(maybe_nssols) # not really a debug code
        if ns_status == MMNoSpaceSolType.imaginary
            debug_code = -4
        else
            debug_code = -5
        end
    else # something odd
        debug_code = -100
        @error (@sprintf "encountered an unrecognized return type of mmp_nospace_sol %s" string(ns_status))
    end

    if found_valid_ss
        for pdci in params_ci_diff
            lDN, lDG, lDR = params_diff[pdci]
            Ds = SA[lDN, lDG, lDR]

            M1 = nospace_stable_ss_M1
            Kp = make_K_polynomial_mm(M1, Ds)
            kroots = find_ks_that_have_nullspace(Kp; threshold)
            k_samples = sample_ks_from_nullspace_ks(kroots)

            has_instability = false
            for k in k_samples
                evals = eigvals!(M1_to_M(M1, Ds, k); sortby=eigen_sortby_reverse)
                if real(evals[1]) > threshold
                    has_instability = true
                    break
                end
            end

            if has_instability
                results_array[pnsci, pdci] = 2
            else
                results_array[pnsci, pdci] = 1
            end
        end
    elseif !isnothing(debug_code)
        if debug_code != 0
            @show debug_code
        end
        results_array[pnsci, :, :, :] .= debug_code
    else
        @show ns_status
        throw(ErrorException("did not find a valid ss but debug was not set either"))
    end
end

function main()
    @time analyze_many_mmps2("./out.jld2";
        m=2 .^ LinRange(-7, 4, 50),
        l=LinRange(0.0, 1.0, 15),
        K=[1.0],
        c=2 .^ LinRange(-7, 10, 80),
        d=2 .^ LinRange(-7, 10, 80),
        DN=10 .^ LinRange(-9, -2, 8),
        DG=[1.0],
        DR=10 .^ LinRange(-7, 5, 40),
        threshold=10 * eps(Float64),
        usenthreads=Threads.nthreads()
    )
end

function ltest(N, fname=nothing; kwargs...)
    @time analyze_many_mmps2(fname;
        m=LinRange(0.1, 2.0, N),
        l=LinRange(0.0, 1.0, N),
        K=LinRange(0.1, 10.0, N),
        c=LinRange(0.01, 20.0, N),
        d=LinRange(0.01, 20.0, N),
        DN=10 .^ LinRange(-5, 3, N),
        DG=10 .^ LinRange(-5, 3, N),
        DR=10 .^ LinRange(-5, 3, N),
        threshold=10 * eps(Float64),
        usenthreads=Threads.nthreads(),
        kwargs...
    )
end
