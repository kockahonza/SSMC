using SSMC
using SSMC.BasicMiCRM

using StaticArrays
using DifferentialEquations
using GLMakie

function make_cosmo2(T=1.0; solveandplot=true)
    func = solveandplot ? make_solve_plot_return : make_micrm_smart

    func(2, 2, T)
end

function make_cosmo3(T; solveandplot=true, kwargs...)
    func = solveandplot ? make_solve_plot_return : make_micrm_smart

    func(2, 3, T;
        u0rand=0.1,
        D=[0.0 1.0 0.5; 1.0 0.0 0.5; 0.0 0.0 0.0],
        K=[0.0, 0.0, 1.0],
        kwargs...
    )
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

function nb_c3_ssensemble()
    Ks = [1.0]
    r1s = [0.0]
    r2s = [0.0]
    rgs = [1.0]
    l1s = [0.0]
    l2s = [0.0]
    lgs = [1.0]
    m1s = [1.0]
    m2s = [1.0]
    c1s = LinRange(1.0, 15.0, 5)
    c2s = LinRange(1.0, 15.0, 5)
    cg1s = LinRange(1.0, 15.0, 5)
    cg2s = LinRange(1.0, 15.0, 5)

    rl, range_cis = setup_ranges(Ks, r1s, r2s, rgs, l1s, l2s, lgs, m1s, m2s, c1s, c2s, cg1s, cg2s)
    numruns = length(range_cis)

    function make_prob(i)
        params = makep_cosmo3direct(rl(i)...)
        u0 = make_micrmu0_smart(params; u0=:onlyN, u0rand=0.01)
        SteadyStateProblem(micrmfunc!, u0, params)
    end
    prob = make_prob(1)

    data = [Array{Float64,ndims(range_cis)}(undef, size(range_cis)) for _ in 1:5]

    @info "Making the ensemble"
    ep = EnsembleProblem(prob;
        prob_func=(_, i, _) -> make_prob(i),
        output_func=(sol, i) -> (sol, false),
        reduction=function (u, data, I)
            for (d, i) in zip(data, I)
                for ui in 1:5
                    u[ui][range_cis[i]] = d.u[ui]
                end
            end

            (u, false)
        end,
        u_init=data,
        safetycopy=false
    )

    @info "Solving the ensemble"
    es = solve(ep, DynamicSS(), EnsembleThreads(); trajectories=numruns, batch_size=30)

    es, rl, range_cis
end

function make_cosmo3_ens(T=100.0;
    Ks=[1.0],
    r1s=[0.0], r2s=[0.0], rgs=[1.0],
    l1s=[0.0], l2s=[0.0], lgs=[1.0],
    m1s=[1.0], m2s=[1.0],
    c1s=[10.0], c2s=[10.0], cg1s=[10.0], cg2s=[10.0],
    output_func=(sol, i) -> (sol, false),
)
    rl, numruns = setup_range_lookup(Ks, r1s, r2s, rgs, l1s, l2s, lgs, m1s, m2s, c1s, c2s, cg1s, cg2s)

    prob = make_cosmo3direct(T, rl(1)...; solveandplot=false)

    EnsembleProblem(prob; output_func,
        prob_func=(_, i, _) -> make_cosmo3direct(T, rl(i)...; solveandplot=false),
        safetycopy=false
    )
end
