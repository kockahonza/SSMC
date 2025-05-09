using SSMCMain, SSMCMain.ModifiedMiCRM.RandomSystems

using Base.Threads, OhMyThreads
using Random, Distributions
using NamedArrays
using FreqTables
using JLD2

function do_single(params, lst, extinctthreshold, maxresidwarn)
    Ns, Nr = get_Ns(params)

    # numerically solve for the steady state
    u0 = ModifiedMiCRM.make_u0_onlyN(params)
    ssp = make_mmicrm_ss_problem(params, u0)
    ssps = solve(ssp, DynamicSS(QNDF()); reltol=maxresidwarn / 1000)

    if SciMLBase.successful_retcode(ssps.retcode)
        warning = false
        maxresid = maximum(abs, ssps.resid)
        if maxresid > maxresidwarn
            @printf "maxresid reached is %g > %g" maxresid maxresidwarn
            warning = true
        end

        if all(x -> abs(x) < extinctthreshold, ssps.u[1:Ns])
            return -101
        end

        linstab_result = lst(params, ssps.u)
        if !warning
            if linstab_result
                return 2 # spatial instability
            else
                return 1 # stable
            end
        else
            if linstab_result
                return -2 # spatial instability but may be wrong
            else
                return -1 # stable but may be wrong
            end
        end

    else
        return -100
    end
end
function interesting_rslt(x::Int)
    x in (-2, 2)
end

function do_rg_run(rg, num_repeats, ks;
    extinctthreshold=1e-7,
    maxresidwarn=1e-7
)
    @time "Generating one params" sample_params = rg()
    flush(stdout)
    Ns, Nr = get_Ns(sample_params)
    N = Ns + Nr

    # prep for the run
    lst = LinstabScanTester(ks, N, 0.0)

    rslts = fill(0, num_repeats)
    int_lock = ReentrantLock()
    interesting_params = Any[]
    @tasks for i in 1:num_repeats
        @local llst = copy(lst)

        params = rg()

        rslts[i] = do_single(params, llst, extinctthreshold, maxresidwarn)

        if interesting_rslt(rslts[i])
            lock(int_lock) do
                push!(interesting_params, params)
            end
        end

        # @printf "Run %d -> %d\n" i rslts[i]
        # flush(stdout)
    end

    rslts, interesting_params
end

function main()
    Ns = 3:20
    num_reps = 10000
    ks = LinRange(0.0, 40.0, 100000)

    codes = [2, -2, 1, -1, -101, -100, 0]

    ncodes = length(codes)
    ftables = NamedArray(Matrix{Int}(undef, length(Ns), ncodes + 1), (Ns, [codes; "other"]), ("N", "code"))

    for (N_i, N) in enumerate(Ns)
        srg = SRGStevens1(N, N, 1.0, 0.35)

        tr = @timed rslts, intps = do_rg_run(srg, num_reps, ks)

        lft = fill(0, length(codes) + 1)
        for r in rslts
            i = findfirst(x -> x == r, codes)
            if !isnothing(i)
                lft[i] += 1
            else
                lft[end] += 1
            end
        end
        ftables[N_i, :] .= lft

        @printf "N=%d finished and took %f seconds\n" N tr.time
        flush(stdout)

        fname = @sprintf "out_N%d.jld2" N
        jldsave(fname; srg, rslts, intps)
    end

    jldsave("out_ft.jld2"; ft=ftables)
    display(ftables)
end

function ltest(Ns=3:5, num_reps=100, save=false; kwargs...)
    ks = LinRange(0.0, 40.0, 10000)

    codes = [2, -2, 1, -1, -101, -100, 0]

    ncodes = length(codes)
    ftables = NamedArray(Matrix{Int}(undef, length(Ns), ncodes + 1), (Ns, [codes; "other"]), ("N", "code"))

    for (N_i, N) in enumerate(Ns)
        srg = SRGStevens1(N, N, 1.0, 0.35)

        tr = @timed rslts, intps = do_rg_run(srg, num_reps, ks; kwargs...)

        lft = fill(0, length(codes) + 1)
        for r in rslts
            i = findfirst(x -> x == r, codes)
            if !isnothing(i)
                lft[i] += 1
            else
                lft[end] += 1
            end
        end
        ftables[N_i, :] .= lft

        @printf "N=%d finished and took %f seconds\n" N tr.time
        flush(stdout)

        if save
            fname = @sprintf "lt_out_N%d.jld2" N
            jldsave(fname; srg, rslts, intps)
        end
    end

    ftables
end
