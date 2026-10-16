include("../base.jl")

make_setup() = make_setup("setup.jld2";
    K=5.0, l=0.999, m=1.0, c=1.0, d=1.0,
    DN=1e-3, DI=1.0,
    ps=[10 .^ range(-1, 0, 5); 2],
    num_reps=12,
    T=1e7,
    sN=5000,
    min_peaks=100,
    epsilon=1e-3,
    ks=10 .^ range(-5, 3, 20000),
    seed=20260823,
)

function main()
    run_pde_setup("setup.jld2", "main", 20;
        abstol=1e-7,
        reltol=1e-9,
        maxtime=60 * 60,
        run_threads=nthreads(),
        solver_threads=nothing,
    )
end
