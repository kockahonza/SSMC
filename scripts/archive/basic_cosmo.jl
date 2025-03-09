using SSMC
using SSMC.BasicMiCRM
using MLSolver

using DataFrames

using GLMakie

function make_cosmo2(T=1.0; solveandplot=true)
    func = solveandplot ? make_solve_plot_return : make_micrm_smart

    func(2, 2, T)
end

################################################################################
# Cosmo like system with 3 resources, the last being glucose
################################################################################
function make_cosmo3(T; kwargs...)
    params = makep_cosmo3direct(; kwargs...)
    u0 = make_micrmu0_smart(params; u0=:onlyN, u0rand=0.1)
    p = ODEProblem(micrmfunc!, u0, (0, T), params)

    s = solve(p)
    fap = lines(s)
    display(fap)
    axislegend(fap.axis)
    p, s
end
function makep_cosmo3direct(
    K,
    r1, r2, rg,
    l1, l2, lg,
    m1, m2,
    c1, c2, cg1, cg2
)
    make_micrmparams_smart(2, 3;
        D=[0.0 1.0 0.5; 1.0 0.0 0.5; 0.0 0.0 0.0],
        K=[0.0, 0.0, K],
        l=[l1, l2, lg],
        r=[r1, r2, rg],
        m=[m1, m2],
        c=[c1 0.0 cg1; 0.0 c2 cg2],
    )
end
function makep_cosmo3direct(;
    K=1.0,
    r1=0.0, r2=0.0, rg=1.0,
    l1=0.0, l2=0.0, lg=1.0,
    m1=1.0, m2=1.0,
    c1=10.0, c2=10.0, cg1=10.0, cg2=10.0,
)
    makep_cosmo3direct(K, r1, r2, rg, l1, l2, lg, m1, m2, c1, c2, cg1, cg2)
end

################################################################################
# Runs an steady states ensemble run
################################################################################
function steadystateensemble()
    Ks = [1.0]
    r1s = [0.0]
    r2s = [0.0]
    rgs = LinRange(1.0, 11.0, 5)
    l1s = [0.0]
    l2s = [0.0]
    lgs = [1.0]
    m1s = LinRange(0.0, 10, 5)
    m2s = LinRange(0.0, 10, 5)
    c1s = LinRange(0.0, 20.0, 5)
    c2s = LinRange(0.0, 20.0, 5)
    cg1s = LinRange(0.0, 20.0, 5)
    cg2s = LinRange(0.0, 20.0, 5)

    rl, range_cis = setup_ranges(Ks, r1s, r2s, rgs, l1s, l2s, lgs, m1s, m2s, c1s, c2s, cg1s, cg2s)
    numruns = length(range_cis)

    function make_prob(i)
        params = makep_cosmo3direct(rl(i)...)
        u0 = make_micrmu0_smart(params; u0=:onlyN, u0rand=0.01)
        SteadyStateProblem(micrmfunc!, u0, params)
    end
    prob = make_prob(1)

    fast_info("Making the ensemble")
    ep = EnsembleProblem(prob;
        prob_func=function (_, i, _)
            make_prob(i)
        end,
        safetycopy=false
    )

    fast_info("Solving the ensemble")
    esraw = solve(ep, DynamicSS(), EnsembleThreads();
        trajectories=numruns,
        maxiters=1e3,
    )

    fast_info("Building df of results")
    df = DataFrame(
        N1=Float64[],
        N2=Float64[],
        R1=Float64[],
        R2=Float64[],
        R3=Float64[],
        retcode=[],
        K=Float64[],
        r1=Float64[],
        r2=Float64[],
        rg=Float64[],
        l1=Float64[],
        l2=Float64[],
        lg=Float64[],
        m1=Float64[],
        m2=Float64[],
        c1=Float64[],
        c2=Float64[],
        cg1=Float64[],
        cg2=Float64[],
    )

    for (run_i, sol) in enumerate(esraw.u)
        push!(df, (sol.u..., sol.retcode, rl(run_i)...))
    end

    if any(x -> x != ReturnCode.Success, df.retcode)
        fast_warn("Some solvers did not succeed")
    end

    df
end
