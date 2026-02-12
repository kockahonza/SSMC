include("../../../scripts/single_influx.jl")

function gendata2()
    fname = joinpath("./gd2_" * timestamp() * ".jld2")

    N = 10
    M = N
    B = 3

    xx = LeakageScale.ltox(0.999)
    xxs = range(0.0, xx, 25)
    lis = LeakageScale.l.(xxs)

    Ks = 10 .^ range(-0.5, 4.0, 25)

    raw_dfs = []
    for li in lis
        @show li
        flush(stdout)
        df, cms = do_df_run(Ks, N;
            M,
            pei=1.0,
            linflux=li,
            cinflux=1.0,
            pe=(B / M),
            l=0.0,
            c=1.0,
            num_byproducts=B,
            num_repeats=100,
            lsks=10 .^ range(-5, 3, 2000),
        )
        push!(raw_dfs, df)
    end

    jldsave(fname;
        N, M, B, lis, Ks, raw_dfs
    )

    fname
end

""" Same as gendata2() but finer sampling """
function gendata3()
    fname = joinpath("./gd3_" * timestamp() * ".jld2")

    N = 10
    M = N
    B = 3

    xx = LeakageScale.ltox(0.999)
    xxs = range(0.0, xx, 50)
    lis = LeakageScale.l.(xxs)

    Ks = 10 .^ range(-0.5, 4.0, 100)

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

""" Same as gendata3() but N=20 """
function gendata4()
    fname = joinpath("./gd4_" * timestamp() * ".jld2")

    N = 20
    M = N
    B = 3

    xx = LeakageScale.ltox(0.999)
    xxs = range(0.0, xx, 30)
    lis = LeakageScale.l.(xxs)

    Ks = 10 .^ range(-0.5, 4.0, 50)

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

"""Somewhat revised - specifically for fig 1 with slightly random m, c etc."""
function gendata5()
    fname = joinpath("./gd5_forfigures_" * timestamp() * ".jld2")

    N = 20
    M = N
    B = 3

    xx = LeakageScale.ltox(0.999)
    xxs = range(0.0, xx, 30)
    lis = LeakageScale.l.(xxs)

    Ks = 10 .^ range(-0.5, 4.0, 50)

    raw_dfs = []
    for li in lis
        @show li
        flush(stdout)
        df, cms = do_df_run(Ks, N;
            M,
            # Slightly random vars
            m=base10_lognormal(0.0, 0.001),
            c=base10_lognormal(0.0, 0.001),
            cinflux=base10_lognormal(0.0, 0.001),
            # Always eat influx with fixed l
            pei=1.0, linflux=li,
            # Eat rest with chance at no l
            pe=(B / M), l=0.0,
            # Rest
            num_byproducts=Binomial(M, B / M),
            # Rest 2
            num_repeats=100,
            lsks=10 .^ range(-5, 3, 2000),
        )
        push!(raw_dfs, df)
    end

    jldsave(fname;
        N, M, B, lis, Ks, raw_dfs
    )

    fname
end
