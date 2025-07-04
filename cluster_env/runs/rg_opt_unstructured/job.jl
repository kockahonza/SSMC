using SSMCMain, SSMCMain.ModifiedMiCRM.RandomSystems

using LinearAlgebra
using Random
using Base.Iterators
using Base.Threads
using Logging

using OhMyThreads
using Distributions
using DimensionalData
using JLD2
using Geppetto
using Optimization
using OptimizationBBO

################################################################################
# Prep
################################################################################
function do_rg_run2(rg, num_repeats, kmax, Nks;
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

function sigma_to_mu_ratio1()
    (2 / 3) / 2.355
end

function do_run1(lm, lc, ll, lsi, lsr, lsb;
    N=5,
    num_repeats=100,
    kmax=100,
    Nks=1000,
    Ds=1e-10,
    Dr=1.0,
    timelimit=10.0,
    disable_log=true,
    min_good_data_ratio=0.9,
    bad_data_val=-1.0,
    kwargs...
)
    total_influx = 1.0 * N # setting E (or E/V)
    Kmean = total_influx / (lsi * N)
    K = (Kmean, Kmean * sigma_to_mu_ratio1())

    rsg = RSGJans1(N, N;
        m=(lm, lm * sigma_to_mu_ratio1()),
        r=1.0, # setting T
        sparsity_influx=lsi,
        K,
        sparsity_resources=lsr,
        sparsity_byproducts=lsb,
        c=(lc, lc * sigma_to_mu_ratio1()),
        l=(ll, ll * sigma_to_mu_ratio1()),
        Ds, Dr # diffusions passed through
    )

    logger_to_use = disable_log ? NullLogger() : current_logger

    raw_results = with_logger(logger_to_use) do
        example_do_rg_run2(rsg, num_repeats, kmax, Nks;
            extinctthr=1e-8,
            maxresidthr=1e-8,
            errmaxresidthr=0.1,
            # solver stuff
            ode_solver=TRBDF2(),
            tol=1e-11,
            timelimit,
            kwargs...
        )
    end

    cm = countmap(raw_results)

    num2s = get(cm, 2, 0)
    good_runs = num2s + get(cm, 1, 0) + get(cm, 101, 0)

    good_ratio = good_runs / length(raw_results)
    if good_ratio < min_good_data_ratio
        @warn "Getting less than $(min_good_data_ratio*100)% good runs!! cm is $cm"
        bad_data_val
    else
        num2s / good_runs
    end
end

function do_opt(u0;
    solver=OptimizationBBO.BBO_generating_set_search(),
    maxtime=300,
    kwargs...
)
    opf = OptimizationFunction((u, p) -> -do_run1(u...; kwargs...))
    oop = OptimizationProblem(opf, u0;
        lb=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ub=[Inf, Inf, 1.0, 1.0, 1.0, 1.0],
    )

    traj_u = []
    traj_o = []
    l = ReentrantLock()

    function cb(st, ov)
        lock(l) do
            @printf "%g - %s\n" st.objective string(st.u)
            flush(stdout)
            push!(traj_u, st.u)
            push!(traj_o, st.objective)
        end
        false
    end

    GC.gc()
    @time s = solve(oop, solver;
        maxtime,
        callback=cb
    )

    s, (traj_u, traj_o)
end

################################################################################
# Main function
################################################################################
u0_1 = [0.9, 2.0, 1.0, 0.2, 0.5, 0.2]

function main_fu0_N10(num_opts=10)
    BLAS.set_num_threads(1)

    sols = []
    trajectories = []

    tempdirname = "temp_" * randname()
    mkdir(tempdirname)

    for i in 1:num_opts
        u0 = u0_1
        @printf "Starting run %d\n" i
        flush(stdout)
        s, t = do_opt(u0;
            solver=OptimizationBBO.BBO_separable_nes(),
            maxtime=60 * 60 * 2,
            timelimit=60,
            num_repeats=1000,
        )
        push!(sols, s)
        push!(trajectories, t)

        tempfname = tempdirname * "/$i.jld2"
        jldsave(tempfname; u0=u0, s=s, t=t)
    end

    jldsave("./fu0_N10.jld2"; sols, trajectories)
end
