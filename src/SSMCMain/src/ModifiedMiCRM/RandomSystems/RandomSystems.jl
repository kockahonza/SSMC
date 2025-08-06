module RandomSystems

using Reexport
@reexport using ..ModifiedMiCRM

using StatsBase
using Distributions
using OhMyThreads
using HypothesisTests


################################################################################
# Example random system runners
################################################################################
function example_do_rg_run(rg, num_repeats, ks;
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
            else
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
export example_do_rg_run

"""
More flexible linear stability implemented directly in this function
"""
function example_do_rg_run2(rg, num_repeats, kmax, Nks;
    maxresidthr=1e-7,            # will warn if ss residues are larger than this
    errmaxresidthr=1e6 * maxresidthr, # if above this will return error code
    extinctthr=maxresidthr / 10, # species below this value are considered extinct
    lszerothr=1000 * eps(),      # values +- this are considered 0 in linstab analysis
    lspeakthr=lszerothr,
    # whether and which params to return for further examination (int <-> interesting)
    return_int=nothing,
    return_int_sss=true,
    # ss solver setup
    ode_solver=TRBDF2(),
    tol=maxresidthr / 1000,
    timelimit=nothing, # time limit for one solver run in seconds
    abstol=tol,
    reltol=tol,
    maxiters=100000,
    # passed to make_mmicrm_ss_problem
    kwargs...
)
    sample_params = rg()
    Ns, Nr = get_Ns(sample_params)
    N = Ns + Nr

    # handle interesting systems setup
    int_func = if isnothing(return_int)
        nothing
    elseif isa(return_int, Vector) || isa(return_int, Tuple)
        c -> c in return_int
    elseif isa(return_int, Function)
        return_int
    elseif return_int == :all
        c -> true
    else
        throw(ArgumentError("return_interesting needs to be either a list of codes or a custom function"))
    end

    solver_kwargs = (; maxiters, abstol, reltol)
    if !isnothing(timelimit)
        solver_kwargs = (; solver_kwargs..., callback=make_timer_callback(timelimit))
    end

    # setup ks for linstab analysis
    ks = LinRange(0.0, kmax, Nks)[2:end] # 0 is handled separately

    # setup the returned data containers
    rslts = fill(0, num_repeats)

    # these may not be used, skipping the if to not have them boxed
    int_lock = ReentrantLock()
    int_systems_to_return = typeof(sample_params)[]
    int_systems_sss = Vector{Float64}[]

    # the core of the function
    @localize solver_kwargs @tasks for i in 1:num_repeats
        # Prealloc variables in each thread (task)
        @local begin
            M1 = Matrix{Float64}(undef, N, N)
            M = Matrix{Float64}(undef, N, N)
            mrls = Vector{Float64}(undef, length(ks))
        end

        # Setup one random system
        params = rg()
        u0 = ModifiedMiCRM.make_u0_onlyN(params)
        ssp = make_mmicrm_ss_problem(params, u0; kwargs...)
        result = 0
        warning = false

        ######################################## 

        # numerically solve for the steady state
        ssps = solve(ssp, DynamicSS(ode_solver); solver_kwargs...)

        # Check the solver
        if !SciMLBase.successful_retcode(ssps.retcode)
            result = -1000 # solver failed return code
            result -= Int(ssps.original.retcode)
            if ssps.original.retcode == ReturnCode.MaxTime
                @warn "Solver quit due to time limit being reached"
                flush(stderr)
            end
            @goto handle_result
        end
        # Check that the steady state is steady enough
        maxresid = maximum(abs, ssps.resid)
        if maxresid > errmaxresidthr
            @warn (@sprintf "maxresid reached is %g which is above the error threshold of %g" maxresid maxresidthr)
            result = -2000 # maxresid is way beyond any reasonable values
            @goto handle_result
        elseif maxresid > maxresidthr
            @warn (@sprintf "maxresid reached is %g > %g" maxresid maxresidthr)
            warning = true
        end

        # Check for a full extinction
        if all(x -> abs(x) < extinctthr, ssps.u[1:Ns])
            result = 101 # gone extinct in nospace ss
            @goto handle_result
        end

        # Do linear stability

        # handle the k=0 case
        make_M1!(M1, params, ssps.u)
        k0mrl = maximum(real, eigvals!(M1))

        # calculate mrls
        make_M1!(M1, params, ssps.u)
        for (ki, k) in enumerate(ks)
            M .= M1
            M1_to_M!(M, get_Ds(params), k)
            evals = eigvals!(M)
            mrls[ki] = maximum(real, evals)
        end

        # evaluate the mrl results
        maxmrl, maxi = findmax(mrls)

        if k0mrl < -lszerothr # this is the ideal case
            if maxmrl < -lspeakthr
                result = 1
                @goto handle_result
            else
                result = 2
                @goto handle_result
            end
        elseif k0mrl < lszerothr # this can happen when there are interchangeable species, or when close to numerical issues
            if maxmrl < -lspeakthr
                result = 11
                @goto handle_result
            else
                is_separated = false
                for intermediate_mrl in mrls[1:maxi]
                    if intermediate_mrl < -lszerothr
                        is_separated = true
                        break
                    end
                end
                if is_separated # k0 is sketchy but we have a separated positive peak
                    result = 12
                    @goto handle_result
                else # the largest peak is connected to a positive k0 - clearly messy
                    result = 13
                    @goto handle_result
                end
            end
        else # something is definitely off here, however still do the same analysis for extra info
            if maxmrl < lspeakthr
                result = 21
                @goto handle_result
            else
                is_separated = false
                for intermediate_mrl in mrls[1:maxi]
                    if intermediate_mrl < -lszerothr
                        is_separated = true
                        break
                    end
                end
                if is_separated
                    result = 22
                    @goto handle_result
                else
                    result = 23
                    @goto handle_result
                end
            end
        end

        ######################################## 

        @label handle_result
        if warning
            result *= -1
        end
        rslts[i] = result
        if !isnothing(int_func) && int_func(result)
            lock(int_lock) do
                push!(int_systems_to_return, params)
                if return_int_sss
                    push!(int_systems_sss, ssps.u)
                end
            end
        end
    end

    if isnothing(int_func)
        rslts
    else
        if !return_int_sss
            rslts, int_systems_to_return
        else
            rslts, int_systems_to_return, int_systems_sss
        end
    end
end
export example_do_rg_run2

################################################################################
# Other util bits
################################################################################
function instability_stats(cm;
    unstable_codes=[2],
    other_good_codes=[101, 1],
    return_confint_only=true
)
    unstable_runs = sum(c -> get(cm, c, 0), unstable_codes)
    other_good_runs = sum(c -> get(cm, c, 0), other_good_codes)

    bt = BinomialTest(unstable_runs, unstable_runs + other_good_runs)

    if return_confint_only
        (bt.x / bt.n), confint(bt; method=:wilson)
    else
        bt
    end
end
function instability_stats(rslt_codes::Vector; kwargs...)
    instability_stats(countmap(rslt_codes); kwargs...)
end
export instability_stats

################################################################################
# Debugging variants
################################################################################
"""
Similar to example_do_rg_run2 but instead plots the dispersion relations
"""
function rg_run_plot_dispersions(rg, num_repeats, kmax, Nks;
    extinctthr=1e-8,
    abstol=1e-8,
    reltol=1e-8,
    maxresidthr=max(abstol, reltol),
    lszerothr=1000 * eps(),
    int_codes=nothing
)
    # test the random system generator
    @time "Generating one params" sample_params = rg()
    flush(stdout)
    Ns, Nr = get_Ns(sample_params)
    N = Ns + Nr

    # setup ks for linstab analysis
    ks = LinRange(0.0, kmax, Nks)[2:end] # 0 is handled separately

    # setup the returned data containers
    rslts = fill(0, num_repeats)
    rslt_mrls = fill(Float64[], num_repeats)
    rslt_maxresids = Vector{Float64}(undef, num_repeats)
    rslt_maxstrain = Vector{Float64}(undef, num_repeats)
    rslt_numstrains = Vector{Int}(undef, num_repeats)
    interesting_systems = []

    @tasks for i in 1:num_repeats
        # Prealloc variables in each thread (task)
        @local begin
            M1 = Matrix{Float64}(undef, N, N)
            M = Matrix{Float64}(undef, N, N)
            mrls = Vector{Float64}(undef, length(ks))
        end

        # Setup one random system
        params = rg()
        u0 = ModifiedMiCRM.make_u0_onlyN(params)
        ssp = make_mmicrm_ss_problem(params, u0)
        result = 0
        warning = false

        ######################################## 

        # numerically solve for the steady state
        ssps = solve(ssp, DynamicSS(QNDF());
            reltol, abstol
        )
        maxresid = maximum(abs, ssps.resid)
        rslt_maxresids[i] = maxresid
        rslt_maxstrain[i] = maximum(ssps.u[1:Ns])
        rslt_numstrains[i] = count(x -> x > extinctthr, ssps.u[1:Ns])

        # Check the solver
        if !SciMLBase.successful_retcode(ssps.retcode)
            result = -100 # solver failed return code
            @goto handle_result
        end
        # Check that the steady state is steady enough
        if maxresid > maxresidthr
            @warn (@sprintf "maxresid reached is %g which is close to %g" maxresid maxresidthr)
            warning = true
        end

        # Check for a full extinction
        if all(x -> abs(x) < extinctthr, ssps.u[1:Ns])
            result = 101 # gone extinct in nospace ss
            @goto handle_result
        end

        # Do linear stability

        # handle the k=0 case
        make_M1!(M1, params, ssps.u)
        k0mrl = maximum(real, eigvals!(M1))

        # calculate mrls
        make_M1!(M1, params, ssps.u)
        for (ki, k) in enumerate(ks)
            M .= M1
            M1_to_M!(M, get_Ds(params), k)
            evals = eigvals!(M)
            mrls[ki] = maximum(real, evals)
        end

        rslt_mrls[i] = vcat(k0mrl, mrls)

        # evaluate the mrl results
        maxmrl, maxi = findmax(mrls)

        if k0mrl < -lszerothr # this is the ideal case
            if maxmrl > lszerothr
                result = 2
                @goto handle_result
            else
                result = 1
                @goto handle_result
            end
        elseif k0mrl < lszerothr # this is likely having numerical issues but still worth looking at
            if maxmrl > lszerothr
                is_separated = false
                for intermediate_mrl in mrls[1:maxi]
                    if intermediate_mrl < -lszerothr
                        is_separated = true
                        break
                    end
                end
                if is_separated # k0 is sketchy but we have a separated positive peak
                    result = 12
                    @goto handle_result
                else # the largest peak is connected to a positive k0 - clearly messy
                    result = 11
                    @goto handle_result
                end
            else
                result = 10
                @goto handle_result
            end
        else # something is definitely off here, however still do the same analysis for extra info
            if maxmrl > lszerothr
                is_separated = false
                for intermediate_mrl in mrls[1:maxi]
                    if intermediate_mrl < -lszerothr
                        is_separated = true
                        break
                    end
                end
                if is_separated
                    result = 22
                    @goto handle_result
                else
                    result = 21
                    @goto handle_result
                end
            else
                result = 20
                @goto handle_result
            end
        end

        ######################################## 

        @label handle_result
        if warning
            result *= -1
        end
        rslts[i] = result
        if !isnothing(int_codes) && (result in int_codes)
            push!(interesting_systems, params)
        end
    end

    fig = Figure()
    ax = Axis(fig[1, 1])

    plot_ks = vcat(0.0, ks)
    for i in 1:num_repeats
        rslt = rslts[i]
        mrls = rslt_mrls[i]
        maxresid = rslt_maxresids[i]
        numstrains = rslt_numstrains[i]
        maxstrain = rslt_maxstrain[i]
        if !isempty(mrls)
            lines!(ax, plot_ks, mrls;
                label=(@sprintf "Run %d -> %d, %d, %.3g, %.3g" i rslt numstrains maxstrain maxresid
                )
            )
        end
    end
    if any(!isempty, rslt_mrls)
        # axislegend(ax)
        Legend(fig[2, 1], ax)
        colsize!(fig.layout, 1, Relative(1))
        rowsize!(fig.layout, 1, Relative(0.6))
    end

    if isnothing(int_codes)
        fig, rslts
    else
        fig, rslts, interesting_systems
    end
end
export rg_run_plot_dispersions

################################################################################
# Sampling generators
################################################################################
include("stevens.jl")

include("jans_first.jl")
include("jans_v2.jl")

end
