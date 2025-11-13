include("./base.jl")

function do_v3_pde_run(
    logKs, ls, T,
    m, c,
    DN, DI, DR,
    N0, epsilon,
    L, sN;
    outfname="v3_run_" * timestamp() * ".jld2",
    solver=QNDF,
    tol=10000 * eps(),
    maxtime=30 * 60,
    kwargs...
)
    dx = L / sN
    u0 = clamp.(reduce(hcat, [[N0, 0.0, 0.0] .+ epsilon .* randn(3) for _ in 1:sN]), 0.0, Inf)

    params = Matrix{Any}(undef, length(logKs), length(ls))
    retcodes = Matrix{ReturnCode.T}(undef, length(logKs), length(ls))
    final_states = Matrix{Matrix{Float64}}(undef, length(logKs), length(ls))
    final_Ts = Matrix{Float64}(undef, length(logKs), length(ls))

    prog = Progress(length(logKs) * length(ls))
    @tasks for i in 1:length(logKs)
        logK = logKs[i]
        for (j, l) in enumerate(ls)
            mmp = MMParams(;
                K=10^logK,
                m,
                c,
                l=l,
                k=0.0,
                d=1.0,
            )
            sps = SASMMiCRMParams(
                mmp_to_mmicrm(mmp),
                SA[DN, DI, DR],
                make_cartesianspace_smart(1; dx),
                # nthreads()
            )
            sp = make_smmicrm_problem(sps, copy(u0), T)

            s = solve(sp, solver();
                dense=false,
                save_everystep=false,
                abstol=tol,
                reltol=tol,
                callback=make_timer_callback(maxtime),
                kwargs...
            )

            params[i, j] = sps
            retcodes[i, j] = s.retcode
            final_states[i, j] = s.u[end]
            final_Ts[i, j] = s.t[end]

            next!(prog)
            flush(stdout)
        end
    end
    finish!(prog)

    jldsave(outfname;
        logKs, ls, T,
        m, c,
        DN, DI, DR,
        N0, epsilon,
        L, sN,
        solver, tol, maxtime,
        params, retcodes, final_states, final_Ts,
    )

    (; params, retcodes, final_states, final_Ts)
end

function add_v3_nospace_run!(fname;
    solver=QNDF,
    tol=nothing,
    maxtime=5 * 60,
    kwargs...
)
    f = jldopen(fname, "r+")

    T = f["T"]
    N0 = f["N0"]
    u0 = [N0, 0.0, 0.0]

    tol_ = something(tol, f["tol"])

    params = f["params"]

    retcodes = Matrix{ReturnCode.T}(undef, size(params)...)
    final_states = Matrix{Vector{Float64}}(undef, size(params)...)
    final_Ts = Matrix{Float64}(undef, size(params)...)

    prog = Progress(length(params))
    @tasks for ci in eachindex(params)
        ps = params[ci].mmicrm_params
        p = make_mmicrm_problem(ps, copy(u0), T)

        s = solve(p, solver();
            abstol=tol_,
            reltol=tol_,
            callback=make_timer_callback(maxtime),
            kwargs...
        )

        retcodes[ci] = s.retcode
        final_states[ci] = s.u[end]
        final_Ts[ci] = s.t[end]
    end
    finish!(prog)

    f["ns_retcodes"] = retcodes
    f["ns_final_states"] = final_states
    f["ns_final_Ts"] = final_Ts

    (; retcodes, final_states, final_Ts)

    close(f)
end

################################################################################
# PDEs version 2 - high N0, higher DN
################################################################################
function v3main_base()
    outfname = "v3_base1.jld2"
    pde_results = do_v3_pde_run(
        range(-0.1, 3, 10),
        LeakageScale.lxrange(0.01, 0.99, 10),
        1e6,
        1.0, 1.0,
        1e-6, 1.0, 1.0,
        100.0, 1e-5,
        5, 5000;
        outfname,
    )
    @show countmap(pde_results.retcodes)
    nospace_results = add_v3_nospace_run!(outfname)
    @show countmap(nospace_results.retcodes)
end

function v3main_lowDR()
    outfname = "v3_lowDR1.jld2"
    pde_results = do_v3_pde_run(
        range(-0.1, 3, 80),
        LeakageScale.lxrange(0.01, 0.99, 30),
        1e6,
        1.0, 1.0,
        1e-6, 1.0, 0.01,
        100.0, 1e-5,
        5, 5000;
        outfname,
    )
    @show countmap(pde_results.retcodes)
    nospace_results = add_v3_nospace_run!(outfname)
    @show countmap(nospace_results.retcodes)
end

function v3main_lowm_lowDR()
    outfname = "v3_lowm_lowDR1.jld2"
    pde_results = do_v3_pde_run(
        range(-2.1, 0.5, 80),
        LeakageScale.lxrange(0.01, 0.99, 30),
        1e6,
        0.01, 1.0,
        1e-6, 1.0, 0.01,
        100.0, 1e-5,
        5, 5000;
        outfname,
    )
    @show countmap(pde_results.retcodes)
    nospace_results = add_v3_nospace_run!(outfname)
    @show countmap(nospace_results.retcodes)
end

function v3main()
    v3main_base()
    v3main_lowDR()
    v3main_lowm_lowDR()
end
