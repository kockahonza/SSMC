using SSMC, SSMC.BasicMiCRM

using Printf
using DataFrames
using JLD2

using Base.Threads
using OhMyThreads

function run_scan_par(; unstable_threshold=0.0)
    df_param_cols, itoparams, paramcis = prep_paramscan(;
        # resources
        K1=[1.0],
        K2=[1.0],
        r1=[1.0],
        r2=[1.0],
        l1=[1.0],
        l2=[1.0],
        # uptake rates
        c11=LinRange(0.5, 20, 2), # uptake rate of 1 of its glucose
        c13=LinRange(0.5, 20, 2), # uptake rate of 1 of its growth resources
        c22=LinRange(0.5, 20, 2), # uptake rate of 2 of its glucose
        c24=LinRange(0.5, 20, 2), # uptake rate of 2 of its growth resources
        # strain death rates
        m1=LinRange(1e-6, 2.0, 2),
        m2=LinRange(1e-6, 2.0, 2),
        # diffusion rates
        Ds1=LinRange(0.0, 5.0, 2),
        Ds2=LinRange(0.0, 5.0, 2),
        Dr1=LinRange(0.0, 100.0, 2),
        Dr2=LinRange(0.0, 100.0, 2),
        Dr3=LinRange(0.0, 100.0, 2),
        Dr4=LinRange(0.0, 100.0, 2),
    )

    ks = LinRange(0.0, 5.0, 10)

    df = DataFrame(;
        Nss1=Float64[],
        Nss2=Float64[],
        Rss1=Float64[],
        Rss2=Float64[],
        Rss3=Float64[],
        Rss4=Float64[],
        ss_retcode=ReturnCode.T[],
        unstable=Union{Nothing,Bool}[],
        lambdas=Union{Nothing,Matrix{ComplexF64}}[],
        df_param_cols...
    )
    metadata!(df, "ks", ks)


    i_chunks = chunks(1:length(paramcis); n=nthreads())
    local_dfs = [copy(df) for _ in 1:length(i_chunks)]

    @sync for (is, local_df) in zip(i_chunks, local_dfs)
        @spawn begin
            numis = length(is)
            rep_vals = floor.([numis * 0.1, numis * 0.3, numis * 0.5, numis * 0.7, numis * 0.9])
            for (progress_i, i) in enumerate(is)
                for rep_val in rep_vals
                    if progress_i == rep_val
                        @printf "thread %d is at index %d out of %d\n" threadid() rep_val numis
                    end
                end

                raw_params = itoparams(i)
                K1, K2, r1, r2, l1, l2, c11, c13, c22, c24, m1, m2, Ds1, Ds2, Dr1, Dr2, Dr3, Dr4 = raw_params

                params = MiCRMParams(
                    SA[1.0, 1.0],
                    SA[1.0, 1.0, 1.0, 1.0],
                    SA[m1, m2],
                    SA[l1, l2, 0.0, 0.0],
                    SA[K1, K2, 0.0, 0.0],
                    SA[r1, r2, 0.0, 0.0],
                    SA[c11 0.0 c13 0.0; 0.0 c22 0.0 c24],
                    SA[0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0]
                )
                u0 = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
                ssp = SteadyStateProblem(micrmfunc!, u0, params)

                sssol = solve(ssp, DynamicSS(); maxiters=1000)
                if sssol.retcode == ReturnCode.Success
                    lambdas = do_linstab_for_ks(ks, params, [Ds1, Ds2, Dr1, Dr2, Dr3, Dr4], sssol.u)
                    unstable = any(x -> real(x) > unstable_threshold, lambdas)
                else
                    lambdas = nothing
                    unstable = nothing
                end

                push!(local_df, (sssol.u..., sssol.retcode, unstable, lambdas, raw_params...))
            end
        end
    end

    df = reduce(vcat, local_dfs)

    if any(df.unstable)
        @info "Found some unstable configs"
    end

    df
end
