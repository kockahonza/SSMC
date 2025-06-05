using SSMCMain, SSMCMain.ModifiedMiCRM.RandomSystems

using LinearAlgebra
using Random
using Base.Iterators
using Base.Threads

using OhMyThreads
using Distributions
using DimensionalData
using JLD2

################################################################################
# Prep
################################################################################
function do_rg_run2(rg, num_repeats, kmax, Nks;
    maxresidthr=1e-7,            # will warn if ss residues are larger than this
    extinctthr=maxresidthr / 10, # species below this value are considered extinct
    lszerothr=1000 * eps(),      # values +- this are considered 0 in linstab analysis
    lspeakthr=lszerothr,
    # whether and which params to return for further examination (int <-> interesting)
    return_int=nothing,
    return_int_sss=true,
    debug_save_problem=nothing,
    # ss solver target tolerances and maxiters
    tol=maxresidthr / 10,
    timelimit=nothing, # time limit for one solver run in seconds
    abstol=tol,
    reltol=tol,
    maxiters=100000,
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

    debug_save_lock = ReentrantLock()

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
        ssp = make_mmicrm_ss_problem(params, u0)

        if !isnothing(debug_save_problem)
            lock(debug_save_lock) do
                fname = debug_save_problem * string(i) * ".jld2"
                save_object(fname, ssp)
            end
        end

        result = 0
        warning = false

        ######################################## 

        # numerically solve for the steady state
        @printf "%d starting solver\n" i
        flush(stdout)
        ssps = solve(ssp, DynamicSS(QNDF()); solver_kwargs...)
        @printf "%d finished solver\n" i
        flush(stdout)

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

        @printf "%d starting linstab\n" i
        flush(stdout)

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
        @printf "%d handling result\n" i
        flush(stdout)
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

function scan_func(func, result_type;
    progress=true,
    async_progress=nothing,
    kwargs...
)
    params_prod = product(values(values(kwargs))...)
    params_size = size(params_prod)
    params_cis = CartesianIndices(params_size)

    num_runs = length(params_cis)
    if progress
        pi = 1
    end
    if !isnothing(async_progress)
        api = 1
        ap_running = true
        ap_task = Task(function ()
            while ap_running
                @printf "Working on run %d out of %d\n" api num_runs
                flush(stdout)
                sleep(async_progress)
            end
        end)
        schedule(ap_task)
    end

    results = Array{result_type}(undef, params_size)
    for (params, ci) in zip(params_prod, params_cis)
        results[ci] = func(params...)

        if progress
            @printf "Just finished run %d out of %d\n" pi num_runs
            pi += 1
            flush(stdout)
        end
        if !isnothing(async_progress)
            api += 1
        end
    end

    if !isnothing(async_progress)
        ap_running = false
        wait(ap_task)
    end

    DimArray(results, (; kwargs...))
end

function sigma_to_mu_ratio1()
    (2 / 3) / 2.355
end

################################################################################
# Different scan setups
################################################################################
function run1(num_repeats=100, kmax=100, Nks=1000;
    N=[5],
    m=[1.0],
    c=[1.0],
    l=[0.5],
    si=[1.0],
    sr=[0.5],
    sb=[1.0],
)
    function func(lN, lm, lc, ll, lsi, lsr, lsb)
        rsg = RSGJans1(lN, lN;
            m=(lm, lm * sigma_to_mu_ratio1()),
            r=1.0, # setting T
            sparsity_influx=lsi,
            K=(1.0, sigma_to_mu_ratio1()), # setting E (or E/V)
            sparsity_resources=lsr,
            sparsity_byproducts=lsb,
            c=(lc, lc * sigma_to_mu_ratio1()),
            l=(ll, ll * sigma_to_mu_ratio1()),
            Ds=1e-8, Dr=1.0, # setting L plus assuming the specific values don't matter as long as Ds << Dr
        )
        raw_results = do_rg_run2(rsg, num_repeats, kmax, Nks;
            maxresidthr=1e-10,
            abstol=1000 * eps(),
            reltol=1000 * eps(),
            maxiters=10000,
        )
        countmap(raw_results)
    end
    scan_func(func, Dict{Int,Int}; N, m, c, l, si, sr, sb)
end

function run2(N, num_repeats=100, kmax=100, Nks=1000;
    m=[1.0],
    c=[1.0],
    l=[0.5],
    si=[1.0],
    sr=[0.5],
    sb=[1.0],
)
    function func(lm, lc, ll, lsi, lsr, lsb)
        rsg = RSGJans1(N, N;
            m=(lm, lm * sigma_to_mu_ratio1()),
            r=1.0, # setting T
            sparsity_influx=lsi,
            K=(1.0, sigma_to_mu_ratio1()), # setting E (or E/V)
            sparsity_resources=lsr,
            sparsity_byproducts=lsb,
            c=(lc, lc * sigma_to_mu_ratio1()),
            l=(ll, ll * sigma_to_mu_ratio1()),
            Ds=1e-8, Dr=1.0, # setting L plus assuming the specific values don't matter as long as Ds << Dr
        )
        raw_results = do_rg_run2(rsg, num_repeats, kmax, Nks;
            maxresidthr=1e-10,
            abstol=1000 * eps(),
            reltol=1000 * eps(),
            maxiters=10000,
        )
        countmap(raw_results)
    end
    scan_func(func, Dict{Int,Int}; m, c, l, si, sr, sb)
end

"""
Does a scan for a single N while keeping the total influx energy rate at 1.0 for
each strain (hopefully valid through dimensional reduction) no matter the rest
of the params.
"""
function run3(N, num_repeats=100, kmax=100, Nks=1000;
    m=[1.0],
    c=[1.0],
    l=[0.5],
    si=[1.0],
    sr=[0.5],
    sb=[1.0],
)
    function func(lm, lc, ll, lsi, lsr, lsb)
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
            Ds=1e-8, Dr=1.0, # setting L plus assuming the specific values don't matter as long as Ds << Dr
        )
        raw_results = do_rg_run2(rsg, num_repeats, kmax, Nks;
            extinctthr=1e-8,
            maxresidthr=1e-8,
            abstol=1000 * eps(),
            reltol=1000 * eps(),
            timelimit=10 * 60.0,
            debug_save_problem="debug_sp/"
        )
        countmap(raw_results)
    end
    scan_func(func, Dict{Int,Int}; m, c, l, si, sr, sb,
        progress=true,
        async_progress=60, # async progress report once every minute
    )
end

################################################################################
# Main function
################################################################################
function main_run1()
    BLAS.set_num_threads(1)
    @time rslts = run1(100, 50, 1000;
        N=4:2:20,
        m=2 .^ range(-5, 5, 6),
        c=2 .^ range(-5, 5, 6),
        l=range(0.0, 1.0, 5),
        si=range(0.0, 1.0, 5),
        sr=range(0.0, 1.0, 5),
        sb=range(0.0, 1.0, 5),
    )
    save_object("./run1_main.jld2", rslts)
    rslts
end

function main_run1_shorter()
    BLAS.set_num_threads(1)
    @time rslts = run1(100, 50, 1000;
        N=[4, 10, 20],
        m=2 .^ range(-3, 3, 3),
        c=2 .^ range(-3, 3, 3),
        l=[0.0, 0.5, 1.0],
        si=range(0.0, 1.0, 5),
        sr=range(0.0, 1.0, 5),
        sb=range(0.0, 1.0, 5),
    )
    save_object("./run1_shorter.jld2", rslts)
    rslts
end

function ltest_run1()
    BLAS.set_num_threads(1)
    @time rslts = run1(100, 50, 100;
        N=[10],
        m=[1.0],
        c=[1.0],
        l=range(0.0, 1.0, 3),
        si=range(0.0, 1.0, 3),
        sr=range(0.0, 1.0, 3),
        sb=range(0.0, 1.0, 3),
    )
    save_object("./run1_ltest.jld2", rslts)
    rslts
end

function main_run2()
    BLAS.set_num_threads(1)
    @time rslts = run2(50, 100, 100.0, 1000;
        m=2 .^ range(-3, 3, 3),
        c=2 .^ range(-3, 3, 3),
        l=range(0.0, 1.0, 4),
        si=range(0.0, 1.0, 4),
        sr=range(0.0, 1.0, 4),
        sb=range(0.0, 1.0, 4),
    )
    save_object("./run2_main.jld2", rslts)
    rslts
end

function main_run3_N10()
    BLAS.set_num_threads(1)
    @time rslts = run3(10, 100, 100.0, 1000;
        m=2 .^ range(-4, 2, 5),
        c=2 .^ range(-2, 6, 5),
        l=range(0.0, 1.0, 4)[1:end],
        si=range(0.0, 1.0, 5)[1:end],
        sr=range(0.0, 1.0, 5)[1:end],
        sb=range(0.0, 1.0, 5)[1:end],
    )
    save_object("./run3_N10.jld2", rslts)
    rslts
end

function ltest_run3_N10()
    BLAS.set_num_threads(1)
    @time rslts = run3(10, 100, 50.0, 100;
        m=2 .^ range(-4, 2, 5),
        c=2 .^ range(-2, 6, 5),
        l=[1.0],
        si=[0.5],
        sr=[0.5],
        sb=[0.5],
    )
    save_object("./ltest_run3_N10.jld2", rslts)
    rslts
end
