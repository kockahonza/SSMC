using SSMCMain, SSMCMain.MinimalModelSemisymbolic

using JLD2
using ChunkSplitters
using Base.Threads

function analyze_many_mmps(save_filename=nothing;
    m=[1.0],
    l=[1.0],
    K=[1.0],
    c=[1.0],
    d=[1.0],
    DN=[0.0],
    DG=[0.0],
    DR=[0.0],
    include_extinct=true,
    threshold=2 * eps(Float64),
    usenthreads=nothing
)
    ns_param_ranges = (m, l, K, c, d)
    ns_cis = CartesianIndices(length.(ns_param_ranges))
    ns_i_to_params = i -> getindex.(ns_param_ranges, ns_cis[i].I)

    diff_ranges = (DN, DG, DR)
    Ds_cis = CartesianIndices(length.(diff_ranges))
    diff_i_to_params = i -> getindex.(diff_ranges, Ds_cis[i].I)

    total_size = (size(ns_cis)..., size(Ds_cis)...)

    # proparties which only depend on the no-space system
    num_ns_steadystates = Array{Int}(undef, size(ns_cis))
    ns_steadystates = Array{Vector{Vector{Float64}}}(undef, size(ns_cis))

    # properties in space (depend on the diffusion constants)
    total_num_modes = Array{Int}(undef, total_size)
    krootss = Array{Vector{Vector{Float64}}}(undef, total_size)
    num_modes_secs = Array{Vector{Vector{Int}}}(undef, total_size)

    if isnothing(usenthreads)
        for ns_ci in ns_cis
            analyze_single_mmps_step_!(
                ns_ci, ns_i_to_params,
                Ds_cis, diff_i_to_params,
                num_ns_steadystates, ns_steadystates,
                total_num_modes, krootss, num_modes_secs,
                include_extinct, threshold
            )
        end
    else
        ns_cis_chunks = chunks(ns_cis; n=usenthreads)
        @sync for ch in ns_cis_chunks
            @spawn begin
                @inbounds for ns_ci in ch
                    analyze_single_mmps_step_!(
                        ns_ci, ns_i_to_params,
                        Ds_cis, diff_i_to_params,
                        num_ns_steadystates, ns_steadystates,
                        total_num_modes, krootss, num_modes_secs,
                        include_extinct, threshold
                    )
                end
            end
        end
    end

    if !isnothing(save_filename)
        jldsave(save_filename;
            m, l, K, c, d,
            DN, DG, DR,
            num_ns_steadystates, ns_steadystates,
            total_num_modes, krootss, num_modes_secs,
            include_extinct, threshold
        )
    end

    num_ns_steadystates, ns_steadystates, total_num_modes, krootss, num_modes_secs
end
function analyze_single_mmps_step_!(
    ns_ci, ns_i_to_params,
    Ds_cis, diff_i_to_params,
    num_ns_steadystates, ns_steadystates,
    total_num_modes, krootss, num_modes_secs,
    include_extinct, threshold
)
    lm, ll, lK, lc, ld = ns_i_to_params(ns_ci)
    mmpns = MinimalModelParamsNoSpace(lm, ll, lK, lc, ld)
    mmicrm_params = mmp_to_mmicrm(mmpns)

    all_steadystates = solve_nospace(mmpns; include_extinct)
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
    sp_steadystates = physical_steadystates[stable_is]
    M1s = M1s_[stable_is]

    # save nospace outputs
    num_ns_steadystates[ns_ci] = length(sp_steadystates)
    ns_steadystates[ns_ci] = sp_steadystates

    # and do the spatial bit
    for Ds_ci in Ds_cis
        lDN, lDG, lDR = diff_i_to_params(Ds_ci)
        Ds = SA[lDN, lDG, lDR]

        l_total_num_modes = 0
        l_krootss = Vector{Float64}[]
        l_num_modes_secs = Vector{Int}[]
        for (ss_i, M1) in enumerate(M1s)
            Kpolynom = make_K_polynomial(M1, Ds)
            Kroots = sort(find_real_positive_cubic_roots(Kpolynom; threshold=threshold))
            kroots = sqrt.(Kroots)

            if length(Kroots) == 0
                k_samples = [1.0]
            elseif length(Kroots) == 1
                k_samples = [kroots[1] / 2, 2 * kroots[1]]
            else
                k_samples = [kroots[1] / 2]

                for i in 2:(length(kroots))
                    push!(k_samples, (kroots[i-1] + kroots[i]) / 2)
                end

                push!(k_samples, 2 * kroots[end])
            end

            num_modes_in_sections = [find_number_growing_modes(M1, k, Ds; threshold) for k in k_samples]

            l_total_num_modes += sum(num_modes_in_sections)
            push!(l_krootss, kroots)
            push!(l_num_modes_secs, num_modes_in_sections)
        end

        total_num_modes[ns_ci, Ds_ci] = l_total_num_modes
        krootss[ns_ci, Ds_ci] = l_krootss
        num_modes_secs[ns_ci, Ds_ci] = l_num_modes_secs
    end
end

function main()
    @time analyze_many_mmps("./out.jld2";
        m=LinRange(0.1, 2.0, 10),
        l=LinRange(0.0, 1.0, 10),
        K=LinRange(0.1, 10.0, 10),
        c=LinRange(0.01, 20.0, 10),
        d=LinRange(0.01, 20.0, 10),
        DN=10 .^ LinRange(-5, 3, 9),
        DG=10 .^ LinRange(-5, 3, 9),
        DR=10 .^ LinRange(-5, 3, 9),
        include_extinct=true,
        threshold=2 * eps(Float64),
        usenthreads=Threads.nthreads()
    )
end

function ltest(N)
    @time analyze_many_mmps(;
        m=LinRange(0.1, 2.0, N),
        l=LinRange(0.0, 1.0, N),
        K=LinRange(0.1, 10.0, N),
        c=LinRange(0.01, 20.0, N),
        d=LinRange(0.01, 20.0, N),
        DN=10 .^ LinRange(-5, 3, N),
        DG=10 .^ LinRange(-5, 3, N),
        DR=10 .^ LinRange(-5, 3, N),
        include_extinct=true,
        threshold=2 * eps(Float64),
        usenthreads=Threads.nthreads()
    )
end
