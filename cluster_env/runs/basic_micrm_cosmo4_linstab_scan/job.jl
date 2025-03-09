include("../../../scripts/lsscan_cosmo4.jl")

function run()
    df = run_scan(;
        # resources
        K1=[1.0],
        K2=[1.0],
        r1=[1.0],
        r2=[1.0],
        l1=[1.0],
        l2=[1.0],
        # uptake rates
        c11=LinRange(0.5, 20, 5), # uptake rate of 1 of its glucose
        c13=LinRange(0.5, 20, 5), # uptake rate of 1 of its growth resources
        c22=LinRange(0.5, 20, 5), # uptake rate of 2 of its glucose
        c24=LinRange(0.5, 20, 5), # uptake rate of 2 of its growth resources
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

    save_object("out_df.jld2", df)
end

function run_test()
    df = run_scan(;
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
        unstable_threshold=0.0
    )

    save_object("out_df.jld2", df)
end
