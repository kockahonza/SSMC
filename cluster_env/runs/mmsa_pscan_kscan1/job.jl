using SSMCMain, SSMCMain.ModifiedMiCRM.MinimalModelSemisymbolic

using Base.Threads
using Base.Iterators
using DataFrames
using JLD2
using ChunkSplitters

function analyze_many_mmps_kscan(ks, save_filename=nothing;
    m=[1.0],
    l=[1.0],
    K=[1.0],
    c=[1.0],
    d=[1.0],
    DN=[0.0],
    DG=[0.0],
    DR=[0.0],
    threshold=10 * eps(Float64),
    usenthreads=nothing
)
    ns_param_ranges = (m, l, K, c, d)
    ns_cis = CartesianIndices(length.(ns_param_ranges))
    ns_i_to_params = ci -> getindex.(ns_param_ranges, ci.I)

    diff_ranges = (DN, DG, DR)
    Ds_cis = CartesianIndices(length.(diff_ranges))
    diff_i_to_params = ci -> getindex.(diff_ranges, ci.I)

    df = DataFrame(;
        m=eltype(m)[],
        l=eltype(l)[],
        K=eltype(K)[],
        c=eltype(c)[],
        d=eltype(d)[],
        DN=eltype(DN)[],
        DG=eltype(DG)[],
        DR=eltype(DR)[],
        numss=Int[],
        hasinstab=Bool[],
        ss_stab=Vector{Bool}[]
    )

    @info "Starting the run"
    flush(stdout)
    if isnothing(usenthreads)
        for ns_ci in ns_cis
            analyze_many_mmps_kscan_step_!(
                df, ks,
                ns_ci, ns_i_to_params,
                Ds_cis, diff_i_to_params,
                threshold
            )
        end
    else
        ns_cis_chunks = chunks(ns_cis; n=usenthreads, split=RoundRobin())
        ldfs = [copy(df) for _ in 1:usenthreads]
        @sync for (ch, ldf) in zip(ns_cis_chunks, ldfs)
            @spawn begin
                @info (@sprintf "thread %d starting" threadid())
                flush(stdout)
                for ns_ci in ch
                    analyze_many_mmps_kscan_step_!(
                        ldf, ks,
                        ns_ci, ns_i_to_params,
                        Ds_cis, diff_i_to_params,
                        threshold
                    )
                end
            end
        end
        df = vcat(ldfs...)
    end

    if !isnothing(save_filename)
        num_systems = prod(length, ns_param_ranges) * prod(length, diff_ranges)

        jldsave(save_filename;
            df, num_systems,
            m, l, K, c, d, DN, DG, DR,
            threshold
        )
    end

    df
end
function analyze_many_mmps_kscan_step_!(
    df, ks,
    ns_ci, ns_i_to_params,
    Ds_cis, diff_i_to_params,
    threshold
)
    lm, ll, lK, lc, ld = ns_i_to_params(ns_ci)
    mmpns = MinimalModelParamsNoSpace(lm, ll, lK, lc, ld)
    mmicrm_params = mmp_to_mmicrm(mmpns)

    all_steadystates = solve_nospace(mmpns; include_extinct=false)
    # do the cheap test first, ignore sss with negative values
    physical_steadystates = all_steadystates[nospace_sol_check_physical.(all_steadystates; threshold=2 * threshold)]

    # construct the M1s which are used both for nospace and space linstab analysis
    M1s_ = make_M1.(Ref(mmicrm_params), physical_steadystates)

    # get the stable, physical steady states
    stable_is = Int[]
    for (M1s_i, M1) in enumerate(M1s_)
        if nospace_sol_check_stable(M1; threshold=threshold)
            push!(stable_is, M1s_i)
        end
    end

    # the stable and physical steady states and their M1 matrices
    sp_steadystates = physical_steadystates[stable_is]
    M1s = M1s_[stable_is]

    # and do the spatial bit if there are any steady states
    if length(M1s) > 0
        M_tmp = copy(M1s[1])
        for Ds_ci in Ds_cis
            lDN, lDG, lDR = diff_i_to_params(Ds_ci)
            Ds = SA[lDN, lDG, lDR]

            ss_stability = Vector{Bool}(undef, length(M1s))
            for (ss_i, M1) in enumerate(M1s)
                stable = true
                for k in ks
                    M_tmp .= M1
                    for i in 1:length(Ds)
                        M_tmp[i, i] -= k^2 * Ds[i]
                    end
                    evals = eigvals!(M_tmp)
                    if any(l -> real(l) > threshold, evals)
                        stable = false
                        break
                    end
                end
                ss_stability[ss_i] = stable
            end
            hasinstab = any(!, ss_stability)

            # maybe save the results
            if hasinstab
                push!(df, (
                    lm, ll, lK, lc, ld,
                    lDN, lDG, lDR,
                    length(sp_steadystates),
                    hasinstab,
                    ss_stability
                ))
            end
        end
    end
end

function main()
    @time fdf = analyze_many_mmps_kscan(LinRange(0.0, 500.0, 10000), "./out_fdf.jld2";
        m=2 .^ LinRange(-7, 2, 30),
        l=LinRange(0.0, 1.0, 10),
        K=[1.0],
        c=2 .^ LinRange(-7, 4, 40),
        d=2 .^ LinRange(-7, 4, 40),
        DN=[0.0, 1e-7, 1e-5, 1e-3, 1e-1, 1, 10],
        DG=[1.0],
        DR=10 .^ LinRange(-7, 5, 40),
        threshold=10 * eps(Float64),
        usenthreads=Threads.nthreads()
    )
    @time finite_df = fdf[.!(iszero.(fdf.DN)), :]
    @time save_object("./out_finite.jld2", finite_df)
end

function ltest(N, fname=nothing; kwargs...)
    @time analyze_many_mmps_kscan(LinRange(0.0, 100.0, 10000), fname;
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
