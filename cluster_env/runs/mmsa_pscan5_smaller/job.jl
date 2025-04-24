using SSMCMain, SSMCMain.MinimalModelSemisymbolic

using DataFrames
using JLD2
using ChunkSplitters
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
        nummodes=Int[]
    )

    @info "Starting the run"
    flush(stdout)
    if isnothing(usenthreads)
        for ns_ci in ns_cis
            analyze_single_mmps2_step_!(
                df,
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
                    analyze_single_mmps2_step_!(
                        ldf,
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
function analyze_single_mmps2_step_!(
    df,
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

    # and do the spatial bit
    for Ds_ci in Ds_cis
        lDN, lDG, lDR = diff_i_to_params(Ds_ci)
        Ds = SA[lDN, lDG, lDR]

        total_nummodes = 0
        for (ss_i, M1) in enumerate(M1s)
            Kp = make_K_polynomial_mm(M1, Ds)
            kroots = find_ks_that_have_nullspace(Kp; threshold)
            k_samples = sample_ks_from_nullspace_ks(kroots)

            num_modes_in_sections = [find_number_growing_modes(M1, k, Ds; threshold) for k in k_samples]

            total_nummodes += sum(num_modes_in_sections)
        end

        # maybe save the results
        if total_nummodes > 0
            push!(df, (
                m=lm, l=ll, K=lK, c=lc, d=ld,
                DN=lDN, DG=lDG, DR=lDR,
                numss=length(sp_steadystates),
                nummodes=total_nummodes
            ))
        end
    end
end

function main()
    @time analyze_many_mmps2("./out.jld2";
        m=2 .^ LinRange(-7, 2, 8),
        l=[1.0, 0.9, 0.5, 0.0],
        K=2 .^ LinRange(-4, 5, 8),
        c=2 .^ LinRange(-7, 4, 8),
        d=2 .^ LinRange(-7, 4, 8),
        DN=10 .^ LinRange(-7, 5, 10),
        DG=10 .^ LinRange(-7, 5, 10),
        DR=10 .^ LinRange(-7, 5, 10),
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
