using SSMCMain, SSMCMain.ModifiedMiCRM.RandomSystems

using Base.Threads, OhMyThreads
using LinearAlgebra
using Random, Distributions
using Base.Iterators
using StatsBase
using JLD2

function do_rg_run(rg, num_repeats, ks;
    extinctthreshold=1e-8,
    maxresidthreshold=1e-9,
    linstabthreshold=100 * eps(),
    return_interesting=false
)
    @time "Generating one params" sample_params = rg()
    flush(stdout)
    Ns, Nr = get_Ns(sample_params)
    N = Ns + Nr

    # prep for the run
    lst = LinstabScanTester(ks, N, linstabthreshold)

    rslts = fill(0, num_repeats)
    interesting_systems = []

    @tasks for i in 1:num_repeats
        @local llst = copy(lst)

        params = rg()

        result = 0
        interesting = false

        # numerically solve for the steady state
        u0 = ModifiedMiCRM.make_u0_onlyN(params)
        ssp = make_mmicrm_ss_problem(params, u0)
        ssps = solve(ssp, DynamicSS(QNDF()); reltol=maxresidthreshold)

        if SciMLBase.successful_retcode(ssps.retcode)
            warning = false
            maxresid = maximum(abs, ssps.resid)
            if maxresid > maxresidthreshold * 100
                @warn (@sprintf "maxresid reached is %g which is close to %g" maxresid (maxresidthreshold * 100))
                warning = true
            end

            if all(x -> abs(x) < extinctthreshold, ssps.u[1:Ns])
                result = -101 # gone extinct in nospace ss
            end

            linstab_result = llst(params, ssps.u)
            if !warning
                if linstab_result
                    result = 2 # spatial instability
                    interesting = true
                else
                    result = 1 # stable
                end
            else
                if linstab_result
                    result = -2 # spatial instability but may be wrong
                else
                    result = -1 # stable but may be wrong
                end
            end
        else
            result = -100
        end

        rslts[i] = result
        if return_interesting && interesting
            push!(interesting_systems, params)
        end

        # @printf "Run %d -> %d\n" i rslts[i]
        # flush(stdout)
    end

    if !return_interesting
        rslts
    else
        rslts, interesting_systems
    end
end

function main()
    num_reps = 1000
    ks = LinRange(0.0, 50.0, 1000)[2:end] # do not include k=0!

    sparsities = LinRange(0.1, 1.0, 10)
    counts = []

    BLAS.set_num_threads(1)

    for (ri, sparsity) in enumerate(sparsities)
        # run
        srg = MarslandSampler1(20, 20;
            SA=5, MA=5,
            sparsity
        )
        tr = @timed rslts = do_rg_run(srg, num_reps, ks;
            extinctthreshold=1e-10,
            maxresidthreshold=1e-9,
            linstabthreshold=1e-9,
            return_interesting=false
        )

        # save raw data
        fname = @sprintf "./out_ri%d.jld2" ri
        jldsave(fname; srg, rslts)

        # save counts
        push!(counts, countmap(rslts))

        @printf "N=%d finished and took %f seconds\n" N tr.time
        flush(stdout)
    end

    codes = sort(unique(keys, counts))
end

function ltest(Ns=3:5, num_reps=100, save=false; kwargs...)
end
