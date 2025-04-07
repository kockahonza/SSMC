using SSMCMain, SSMCMain.MinimalModelSemisymbolic

using Base.Threads

function main()
    @time analyze_many_mmps("./out.jld2";
        m=LinRange(0.1, 2.0, 100),
        K=LinRange(0.1, 10.0, 100),
        c=LinRange(0.01, 10., 10),
        d=[2.0],
        DN=[1e-5],
        DG=[500.0],
        DR=[0.1],
        include_extinct=true,
        threshold=2 * eps(Float64),
        usenthreads=Threads.nthreads()
    )
end
