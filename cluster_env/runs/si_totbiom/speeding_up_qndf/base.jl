using Revise
includet("../base.jl")


function main1()
    f = jldopen("./sys1.jld2")
    gen_ps = f["gen_ps"]
    ode_fs = f["ode_fs"]

    ps = 10 .^ range(0, -2, 8)
    ps = ps[[1,3]]

    T = 1e8
    L = 5
    sN = 1000
    sp_epsilon = 1e-3

    save_step = 200

    df = run_1_system_changing_p(
        "data1_larger_abstol.jld2", gen_ps, ode_fs, ps,
        T, L, sN,
        sp_epsilon,
        save_step;
        tol=1e-9,
        abstol=1e-7,
        maxtime=6 * 60 * 60,
        run_threads=2,
        solver_threads=16,
        solver=QNDF,
        do_printing=true,
    )

    @show df.realtimes df.maxresids last.(df.sol_ts)
end

function main2()
    f = jldopen("./sys1.jld2")
    gen_ps = f["gen_ps"]
    ode_fs = f["ode_fs"]

    ps = 10 .^ range(0, -2, 8)

    T = 1e8
    L = 5
    sN = 1000
    sp_epsilon = 1e-3

    save_step = 10

    df = run_1_system_changing_p(
        "data2.jld2", gen_ps, ode_fs, ps,
        T, L, sN,
        sp_epsilon,
        save_step;
        tol=1e-9,
        abstol=1e-7,
        maxtime=6 * 60 * 60,
        run_threads=8,
        solver_threads=1,
        solver=QNDF,
        do_printing=true,
    )

    @show df.realtimes df.maxresids last.(df.sol_ts)
end
