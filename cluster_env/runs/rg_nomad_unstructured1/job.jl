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
import NOMAD

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
        do_rg_run2(rsg, num_repeats, kmax, Nks;
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
        bad_data_val, cm
    else
        num2s / good_runs, cm
    end
end

function do_opt(u0;
    maxtime=300, # in seconds
    granularity=0.0,
    kwargs... # notably N, num_repeats and timelimit
)
    traj = []
    tl = ReentrantLock()

    function nomad_func(u)
        rslt, cm = do_run1(u...; kwargs...)

        lock(tl) do
            push!(traj, (copy(u), rslt, cm))
        end

        (true, true, -rslt) # - as we want to maximize not minimize
    end

    maxval = Inf
    np = NOMAD.NomadProblem(6, 1, ["OBJ"], nomad_func;
        lower_bound=fill(0.0, 6),
        upper_bound=[maxval, maxval, 1.0, 1.0, 1.0, 1.0],
        granularity=fill(granularity, 6)
    )
    np.options.max_time = maxtime
    np.options.display_all_eval = true
    np.options.display_stats = ["OBJ", "SOL", "BBE", "BBO"]

    GC.gc()
    @time s = NOMAD.solve(np, u0)

    s, traj
end

################################################################################
# Main function
################################################################################
u0_1 = [0.9, 2.0, 1.0, 0.2, 0.5, 0.2]

function main_N10_comprehensive1(;
    N=10,
    num_starts=10,
    prescreen=20,
    maxtime=60 * 60 * 1,
    num_repeats=1000,
    timelimit=5,
)
    BLAS.set_num_threads(1)

    u0s = []
    sols = []
    trajectories = []
    pre_ts = []

    tempdirname = "temp_N$(N)_" * randname()
    @printf "Using tempdir %s\n" string(tempdirname)
    mkdir(tempdirname)

    maxval = 10.0
    for i in 1:num_starts
        pre_t = []
        if !isnothing(prescreen)
            found_u0 = false
            for _ in 1:10
                u0 = [maxval * rand(), maxval * rand(), rand(), rand(), rand(), rand()]
                rslt, cm = do_run1(u0...; N, num_repeats, timelimit)
                push!(pre_t, (copy(u0), rslt, cm))
                if rslt > 0.0
                    found_u0 = true
                    break
                end
            end
            if !found_u0
                @warn "Could not find a satisfactory u0 during prescreening"
            end
        else
            u0 = [maxval * rand(), maxval * rand(), rand(), rand(), rand(), rand()]
        end
        push!(pre_ts, pre_t)

        @printf "Starting run %d\n" i
        flush(stdout)
        s, t = do_opt(u0;
            N,
            maxtime,
            num_repeats,
            timelimit,
        )

        tempfname = tempdirname * "/$i.jld2"
        jldsave(tempfname; u0=u0, s=s, t=t, pre_t)
        push!(u0s, u0)
        push!(sols, s)
        push!(trajectories, t)

        @printf "Finished run %d %s -> %s\n" i string(u0) string(s.x_best_feas)
    end

    jldsave("./N$(N)_c1.jld2"; u0s, sols, trajectories, pre_ts)
end

function main_N20_comprehensive1(;
    kwargs...
)
    main_N10_comprehensive1(; N=20, timelimit=10, kwargs...)
end

################################################################################
# Plotting
################################################################################
function plot_trajectory(traj;
    pnames=nothing,
    add_colorbar=true,
    vs_coloring=:time,
    kwargs...
)
    prep = prep_traj_plot(length(traj[1][1]), pnames)
    faa = plot_trajectory!(prep, traj; vs_coloring, kwargs...)

    if add_colorbar
        Colorbar(faa.figure[:, prep.num_ps+1], faa.obj.vs_plots[2, 1];
            label="vs plots - " * string(vs_coloring)
        )
    end

    faa
end

function plot_all_trajectories_obj(ts;
    kwargs...
)
    fake_t = reduce(vcat, ts)
    plot_trajectory(fake_t; vs_coloring=:obj, kwargs...)
end

function plot_trajectory!(prep, traj;
    vs_coloring=:time,
    vs_kwargs=(;),
    single_y=:obj,
    single_coloring=nothing,
    single_kwargs=(;),
)
    vs_plots = Dict{Tuple{Int,Int},Any}()
    for i in 1:prep.num_ps
        for j in 1:(i-1)
            ax = prep.vs_axs[(i, j)]

            xs = []
            ys = []

            for (u, o) in traj
                push!(xs, u[j])
                push!(ys, u[i])
            end

            cs = if vs_coloring == :time
                1:length(traj)
            elseif vs_coloring == :obj
                getindex.(traj, 2)
            end

            sc = scatter!(ax, xs, ys; color=cs, vs_kwargs...)
            vs_plots[(i, j)] = sc
        end
    end

    single_plots = []
    for i in 1:prep.num_ps
        ax = prep.single_axs[i]

        xs = getindex.(getindex.(traj, 1), i)
        if single_y == :time
            ys = 1:length(traj)
            ylabel = "time"
        elseif single_y == :obj
            ys = getindex.(traj, 2)
            ylabel = "objective"
        end

        sc_kwargs = Dict()
        if single_coloring == :time
            sc_kwargs[:color] = 1:length(traj)
            clabel = "time"
        elseif single_coloring == :obj
            sc_kwargs[:color] = getindex.(traj, 2)
            clabel = "objective"
        elseif isnothing(single_coloring)
            clabel = nothing
        end

        sc = scatter!(ax, xs, ys; sc_kwargs...)
        push!(single_plots, sc)

        ax.ylabel = ylabel
        if !isnothing(clabel)
            ax.title = ax.title[] * ", color ~ " * clabel
        end
    end

    FigureAxisAnything(prep.fig, (; prep.single_axs, prep.vs_axs), (; vs_plots))
end

function prep_traj_plot(num_ps, pnames=nothing)
    if isnothing(pnames)
        pnames = ["p$i" for i in 1:num_ps]
    end

    # setup axes
    fig = Figure(size=(1000, 1000))

    single_axs = []
    for i in 1:num_ps
        push!(single_axs, Axis(fig[i, i]; title=pnames[i]))
    end

    vs_axs = Dict{Tuple{Int,Int},Any}()
    for i in 1:num_ps
        for j in 1:(i-1)
            ax = vs_axs[(i, j)] = Axis(
                fig[i, j];
                # title=string((i, j))
            )
            linkxaxes!(ax, single_axs[j])
        end
    end
    for i in 1:num_ps
        for j in 1:(i-1)
            linkyaxes!(vs_axs[(i, j)], vs_axs[(i, 1)])
        end
    end

    # axis labels
    # single_axs[1].ylabel = pnames[1]
    # single_axs[end].xlabel = pnames[end]
    for i in 2:num_ps
        vs_axs[(i, 1)].ylabel = pnames[i]
    end
    for i in 1:(num_ps-1)
        vs_axs[(num_ps, i)].xlabel = pnames[i]
    end

    (; fig, single_axs, vs_axs, pnames, num_ps)
end
