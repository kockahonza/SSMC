include("./base.jl")
include("../../../scripts/mm_Kl_pds.jl")

using Peaks

################################################################################
# Running the PDEs/ODEs
################################################################################
function get_siny_u0(
    sN, dx,
    meanN0, numwaves, waveampfactor,
)
    u0 = fill(0.0, 3, sN)
    N = @view u0[1, :]

    N .= meanN0
    add_1d_many_sines!(N, numwaves, waveampfactor * meanN0 / numwaves, dx)
    clamp!(u0, 0.0, Inf)

    u0
end

function do_v4_pde_run(
    logKs, ls, T,
    m, c,
    DN, DI, DR,
    meanN0, numwaves, waveampfactor,
    L, sN;
    outfname="v4_run_" * timestamp() * ".jld2",
    solver=QNDF,
    tol=10000 * eps(),
    maxtime=30 * 60,
    kwargs...
)
    dx = L / sN

    params = Matrix{Any}(undef, length(logKs), length(ls))
    u0s = Matrix{Matrix{Float64}}(undef, length(logKs), length(ls))
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
            u0 = get_siny_u0(sN, dx, meanN0, numwaves, waveampfactor)
            sp = make_smmicrm_problem(sps, u0, T)

            s = solve(sp, solver();
                dense=false,
                save_everystep=false,
                abstol=tol,
                reltol=tol,
                callback=make_timer_callback(maxtime),
                kwargs...
            )

            params[i, j] = sps
            u0s[i, j] = u0
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
        meanN0, numwaves, waveampfactor,
        L, sN,
        solver, tol, maxtime,
        params, u0s, retcodes, final_states, final_Ts,
    )

    (; params, u0s, retcodes, final_states, final_Ts)
end

function add_v4_nospace_run!(fname;
    solver=QNDF,
    tol=nothing,
    maxtime=5 * 60,
    kwargs...
)
    f = jldopen(fname, "r+")

    T = f["T"]

    tol_ = something(tol, f["tol"])

    params = f["params"]
    space_u0s = f["u0s"]

    retcodes = Matrix{ReturnCode.T}(undef, size(params)...)
    u0s = Matrix{Vector{Float64}}(undef, size(params)...)
    final_states = Matrix{Vector{Float64}}(undef, size(params)...)
    final_Ts = Matrix{Float64}(undef, size(params)...)

    prog = Progress(length(params))
    @tasks for ci in eachindex(params)
        ps = params[ci].mmicrm_params

        N0 = mean(space_u0s[ci][1, :])
        u0 = [N0, 0.0, 0.0]

        p = make_mmicrm_problem(ps, copy(u0), T)

        s = solve(p, solver();
            abstol=tol_,
            reltol=tol_,
            callback=make_timer_callback(maxtime),
            kwargs...
        )

        retcodes[ci] = s.retcode
        u0s[ci] = u0
        final_states[ci] = s.u[end]
        final_Ts[ci] = s.t[end]
    end
    finish!(prog)

    f["ns_retcodes"] = retcodes
    f["ns_u0s"] = u0s
    f["ns_final_states"] = final_states
    f["ns_final_Ts"] = final_Ts

    close(f)

    (; retcodes, u0s, final_states, final_Ts)
end

################################################################################
# Specifying the runs we want to do
################################################################################
function v4main_base()
    outfname = "v4_base1.jld2"
    pde_results = do_v4_pde_run(
        range(-0.1, 3, 100),
        LeakageScale.lxrange(0.01, 0.99, 30),
        1e6,
        1.0, 1.0,
        1e-6, 1.0, 1.0,
        10.0, 100, 10.0,
        5, 5000;
        outfname,
    )
    @show countmap(pde_results.retcodes)
    nospace_results = add_v4_nospace_run!(outfname)
    @show countmap(nospace_results.retcodes)
end
