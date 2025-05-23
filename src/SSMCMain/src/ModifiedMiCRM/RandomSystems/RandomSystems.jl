module RandomSystems

using Reexport
@reexport using ..ModifiedMiCRM

using StatsBase
using Distributions
using OhMyThreads


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

function example_do_rg_run2(rg, num_repeats, kmax, Nks;
    extinctthr=1e-8,
    maxresidthr=1e-7,
    lszerothr=1000 * eps(),
    lsk0maxmrlthr=lszerothr,
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
        ssps = solve(ssp, DynamicSS(QNDF()); reltol=(maxresidthr / 1000))

        # Check the solver
        if !SciMLBase.successful_retcode(ssps.retcode)
            result = -100 # solver failed return code
            @goto handle_result
        end
        # Check that the steady state is steady enough
        maxresid = maximum(abs, ssps.resid)
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
        elseif k0mrl < lsk0maxmrlthr # this is likely having numerical issues but still worth looking at
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

    if isnothing(int_codes)
        rslts
    else
        rslts, interesting_systems
    end
end
export example_do_rg_run2


function rg_run_plot_dispersions(rg, num_repeats, kmax, Nks;
    extinctthr=1e-8,
    abstol=1e-8,
    reltol=1e-8,
    maxresidthr=max(abstol, reltol),
    lszerothr=1000 * eps(),
    lsk0maxmrlthr=lszerothr,
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
        elseif k0mrl < lsk0maxmrlthr # this is likely having numerical issues but still worth looking at
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
    end

    fig = Figure()
    ax = Axis(fig[1, 1])

    plot_ks = vcat(0.0, ks)
    for i in 1:num_repeats
        rslt = rslts[i]
        mrls = rslt_mrls[i]
        maxresid = rslt_maxresids[i]
        if !isempty(mrls)
            lines!(ax, plot_ks, mrls, label=(@sprintf "Run %d -> %d, %.2g" i rslt maxresid))
        end
    end
    if any(!isempty, rslt_mrls)
        # axislegend(ax)
        Legend(fig[1, 2], ax)
    end

    fig, rslts
end
export rg_run_plot_dispersions


################################################################################
# Sampling generators
################################################################################
include("stevens.jl")

include("jans_first.jl")

end
