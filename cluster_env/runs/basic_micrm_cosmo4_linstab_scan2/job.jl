using SSMCMain, SSMCMain.BasicMiCRM

using Printf
using Base.Threads
using Logging

using OhMyThreads
using DataFrames
using JLD2

function run_scan(;
    # resources
    K1=[1.0],
    K2=[1.0],
    r1=[1.0],
    r2=[1.0],
    r3=[1.0],
    r4=[1.0],
    l1=[1.0],
    l2=[1.0],
    # uptake rates
    c11=[1.0], # uptake rate of 1 of its glucose
    c13=[1.0], # uptake rate of 1 of its growth resources
    c22=[1.0], # uptake rate of 2 of its glucose
    c24=[1.0], # uptake rate of 2 of its growth resources
    # strain death rates
    m1=[1.0],
    m2=[1.0],
    # diffusion rates
    Ds1=[0.0],
    Ds2=[0.0],
    Dr1=[1.0],
    Dr2=[1.0],
    Dr3=[1.0],
    Dr4=[1.0],
    ks=LinRange(0.0, 5.0, 100),
    unstable_threshold=0.0,
    disable_warnings=true
)
    df_param_cols, itoparams, paramcis = prep_paramscan(;
        K1, K2, r1, r2, r3, r4, l1, l2,
        c11, c13, c22, c24,
        m1, m2,
        Ds1, Ds2, Dr1, Dr2, Dr3, Dr4
    )

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

    i_chunks = chunks(1:length(paramcis); n=nthreads())
    local_dfs = [copy(df) for _ in 1:length(i_chunks)]

    if disable_warnings
        disable_logging(Warn)
    end

    println("Starting a scan")
    flush(stdout)

    @sync for (is, local_df) in zip(i_chunks, local_dfs)
        @spawn begin
            numis = length(is)
            rep_vals = floor.([numis * 0.1, numis * 0.3, numis * 0.5, numis * 0.7, numis * 0.9])
            for (progress_i, i) in enumerate(is)
                for rep_val in rep_vals
                    if progress_i == rep_val
                        @printf "thread %d is at index %d out of %d\n" threadid() rep_val numis
                        flush(stdout)
                    end
                end

                raw_params = itoparams(i)
                K1, K2, r1, r2, r3, r4, l1, l2, c11, c13, c22, c24, m1, m2, Ds1, Ds2, Dr1, Dr2, Dr3, Dr4 = raw_params

                params = MiCRMParams(
                    SA[1.0, 1.0],
                    SA[1.0, 1.0, 1.0, 1.0],
                    SA[m1, m2],
                    SA[l1, l2, 0.0, 0.0],
                    SA[K1, K2, 0.0, 0.0],
                    SA[r1, r2, r3, r4],
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

    println("Scan finished")
    flush(stdout)

    if disable_warnings
        disable_logging(Debug)
    end

    println("Making df")
    flush(stdout)

    df = reduce(vcat, local_dfs)
    metadata!(df, "ks", ks)

    if any(x -> x == true, df.unstable)
        @info "Found some unstable configs"
    end

    df
end

function run()
    df = run_scan(;
        # resources
        K1=[1.0],
        K2=[1.0],
        r1=[1.0],
        r2=[1.0],
        r3=LinRange(0.0, 2.0, 5),
        r4=LinRange(0.0, 2.0, 5),
        l1=[1.0],
        l2=[1.0],
        # uptake rates
        c11=LinRange(0.5, 20, 3), # uptake rate of 1 of its glucose
        c13=LinRange(0.5, 20, 3), # uptake rate of 1 of its growth resources
        c22=LinRange(0.5, 20, 3), # uptake rate of 2 of its glucose
        c24=LinRange(0.5, 20, 3), # uptake rate of 2 of its growth resources
        # strain death rates
        m1=LinRange(1e-6, 2.0, 3),
        m2=LinRange(1e-6, 2.0, 3),
        # diffusion rates
        Ds1=LinRange(0.0, 5.0, 3),
        Ds2=LinRange(0.0, 5.0, 3),
        Dr1=LinRange(0.0, 100.0, 5),
        Dr2=LinRange(0.0, 100.0, 5),
        Dr3=LinRange(0.0, 100.0, 5),
        Dr4=LinRange(0.0, 100.0, 5),
        unstable_threshold=0.0
    )

    println("Saving df")
    flush(stdout)

    save_object("out_df.jld2", df)

    println("Finished!")
end

function run_test()
    df = run_scan(;
        # resources
        K1=[1.0],
        K2=[1.0],
        r1=[1.0],
        r2=[1.0],
        r3=LinRange(0.0, 1.0, 2),
        r4=LinRange(0.0, 1.0, 2),
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
        unstable_threshold=0.0
    )

    println("Saving df")
    flush(stdout)

    save_object("out_df_test.jld2", df)

    println("Finished!")
end
