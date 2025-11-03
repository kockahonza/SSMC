include("../../../scripts/single_influx.jl")

function gendata2()
    fname = joinpath("./gd2_" * timestamp() * ".jld2")

    N = 10
    M = N
    B = 3

    xx = LeakageScale.ltox(0.999)
    xxs = range(-xx, xx, 25)
    lis = LeakageScale.l.(xxs)

    Ks = 10 .^ range(-0.5, 4.0, 25)

    raw_dfs = []
    for li in lis
        @show li
        flush(stdout)
        df, cms = do_df_run(Ks, N;
            M, pei=1.0,
            linflux=li,
            cinflux=1.0, pe=(B / M),
            l=0.0,
            c=1.0, num_byproducts=B, num_repeats=100,
            lsks=10 .^ range(-5, 3, 2000),
        )
        push!(raw_dfs, df)
    end

    jldsave(fname;
        N, M, B, lis, Ks, raw_dfs
    )

    fname
end

