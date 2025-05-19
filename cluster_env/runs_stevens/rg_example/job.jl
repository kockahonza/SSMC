using SSMCMain, SSMCMain.ModifiedMiCRM.RandomSystems

using Base.Threads, OhMyThreads
using Random, Distributions
using NamedArrays
using FreqTables
using JLD2

# This is copied from RandomSystems.jl so that it can be modified if needed
# but in the future you can check there in case I've improved it in some way
# or another
function do_rg_run(rg, num_repeats, ks;
    extinctthreshold=1e-8,
    maxresidthreshold=1e-9
)
    @time "Generating one params" sample_params = rg()
    flush(stdout)
    Ns, Nr = get_Ns(sample_params)
    N = Ns + Nr

    # prep for the run
    lst = LinstabScanTester(ks, N, 0.0)

    rslts = fill(0, num_repeats)
    @tasks for i in 1:num_repeats
        @local llst = copy(lst)

        params = rg()

        result = 0

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

        # @printf "Run %d -> %d\n" i rslts[i]
        # flush(stdout)
    end

    rslts
end

# This function will be ran on the cluster
function main()
    num_reps = 10000
    ks = LinRange(0.0, 40.0, 10000)

    srg = SRGStevens1(10, 10, 1.0, 0.35)

    results = do_rg_run(srg, num_reps, ks;)
        extinctthreshold=1e-8,
        maxresidthreshold=1e-9
    )

    frequencies = freqtable(results)
    display(frequencies)

    jldsave("out.jld2"; results, frequencies)
end

# I often have these as a smaller version to test run on laptop to see if things work
function ltest(Ns=3:5, num_reps=100, save=false; kwargs...)
    num_reps = 100
    ks = LinRange(0.0, 40.0, 100)

    srg = SRGStevens1(10, 10, 1.0, 0.35)

    results = do_rg_run(srg, num_reps, ks;)
        extinctthreshold=1e-8,
        maxresidthreshold=1e-9
    )

    frequencies = freqtable(results)
    display(frequencies)

    jldsave("out_lt.jld2"; results, frequencies)
end
