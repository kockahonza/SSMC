using SSMCMain, SSMCMain.ModifiedMiCRM.RandomSystems

# using LinearAlgebra
# using Random
# using Base.Iterators
# using Base.Threads

using Logging

using OhMyThreads
using Distributions
using JLD2
using Geppetto
import NOMAD
using DataFrames

function calc_score(cm, top=[2], bottom=[1, 2, 101])
    sum(x -> get(cm, x, 0.0), top) / sum(x -> get(cm, x, 0.0), bottom)
end

function do_run_unimodal_c(
    N,
    me, mev,
    Dse, Dsev, Dre, Drev,
    la, lb, ce, cev, sr, sb,
    K;
    num_repeats=100,
    kmax=50,
    Nks=1000,
    disable_log=true,
    # analyzing result codes
    min_good_data_ratio=0.9,
    bad_data_val=-1.0,
    score_func=calc_score,
    timelimit=nothing
)
    rsg = RSGJans1(N, N;
        m=base10_lognormal(me, mev),
        Ds=base10_lognormal(Dse, Dsev),
        Dr=base10_lognormal(Dre, Drev),
        # use a unimodal c dist
        l=Beta(la, lb),
        c=base10_lognormal(ce, cev),
        sparsity_resources=sr,
        sparsity_byproducts=sb,
        # fix a single influx resource
        num_influx_resources=Dirac(1),
        K=Dirac(K),
    )

    logger_to_use = disable_log ? NullLogger() : current_logger
    result_codes = with_logger(logger_to_use) do
        example_do_rg_run2(rsg, num_repeats, kmax, Nks; timelimit)
    end

    cm = countmap(result_codes)

    good_codes = sum(x -> get(cm, x, 0.0), [1, 2, 101])
    good_ratio = good_codes / num_repeats

    score = score_func(cm)

    if good_ratio < min_good_data_ratio
        @warn "Getting less than $(min_good_data_ratio*100)% good runs!! cm is $cm"
        if !isnothing(bad_data_val)
            score = bad_data_val
        end
    end

    score, cm
end

function main1(;
    N=10,
    num_starts=5,
    num_prescreens=50,
    num_repeats=1000,
    max_nomad_time=60 * 60 * 1,
    max_single_solver_time=20,
    # granularity=1e-3,
)
    BLAS.set_num_threads(1)

    function do_run(u0)
        do_run_unimodal_c(N, u0..., 1.0;
            num_repeats,
            timelimit=max_single_solver_time,
        )
    end

    function gen_u0()
        [
            rand(Uniform(-2.0, 2.0)), 0.0, # m
            -12.0, 0.0, 0.0, 0.0, # diffs 
            1.5 * rand(), 1.5 * rand(), # l
            rand(Uniform(-1.0, 2.0)), 0.0, # c
            rand(), rand(), # sparsities
        ]
    end

    trajectories = []

    for i in 1:num_starts
        traj = []

        # Prescreening
        @info "Prescreening $i"
        flush(stdout)
        if !isnothing(num_prescreens)
            found_u0 = false
            for _ in 1:num_prescreens
                u0 = gen_u0()
                score, cm = do_run(u0)
                if score > 0.0
                    found_u0 = true
                    break
                end
                push!(traj, (copy(u0), score, cm))
            end
            if !found_u0
                @warn "Could not find a satisfactory u0 during prescreening"
            end
        else
            u0 = gen_u0()
        end

        # NOMAD run6
        @info "Starting NOMAD run $i"
        flush(stdout)
        function nomad_func(u)
            score, cm = do_run(u0)

            push!(traj, (copy(u), score, cm))

            (true, true, -score) # - as we want to maximize not minimize
        end

        max1 = 10.0
        max2 = 100.0
        max3 = 100.0

        np = NOMAD.NomadProblem(12, 1, ["OBJ"], nomad_func;
            lower_bound=[
                -max1, 0.0,
                -max2, 0.0, -max2, 0.0,
                0.0, 0.0,
                -max1, 0.0,
                0.0, 0.0
            ],
            upper_bound=[
                max1, Inf,
                max2, Inf, max2, Inf,
                max3, max3,
                max1, Inf,
                1.0, 1.0
            ],
            # granularity=fill(granularity, 12)
        )
        np.options.max_time = max_nomad_time
        np.options.display_all_eval = true
        np.options.display_stats = ["OBJ", "SOL", "BBE", "BBO"]

        GC.gc()
        @time s = NOMAD.solve(np, u0)

        push!(trajectories, traj)
        @info "Finished run $i"
        flush(stdout)
    end

    save_object("N$(N)_unic_" * randname() * ".jld2", trajectories)
    trajectories
end

"""
Same as main1 but fixed diffusions and m and c have no variance
"""
function main2_simpler(;
    N=10,
    num_starts=5,
    num_prescreens=50,
    num_repeats=1000,
    max_nomad_time=60 * 60 * 1,
    max_single_solver_time=20,
    # granularity=1e-3,
)
    BLAS.set_num_threads(1)

    function do_run(u0)
        me, la, lb, ce, sr, sb = u0
        do_run_unimodal_c(N,
            me, 0.0,
            -12.0, 0.0, 0.0, 0.0,
            la, lb,
            ce, 0.0,
            sr, sb,
            num_repeats,
            timelimit=max_single_solver_time,
        )
    end

    function gen_u0()
        [
            rand(Uniform(-2.0, 2.0)), # m
            1.5 * rand(), 1.5 * rand(), # l
            rand(Uniform(-1.0, 2.0)), # c
            rand(), rand(), # sparsities
        ]
    end

    trajectories = []

    for i in 1:num_starts
        traj = []

        # Prescreening
        @info "Prescreening $i"
        flush(stdout)
        if !isnothing(num_prescreens)
            found_u0 = false
            for _ in 1:num_prescreens
                u0 = gen_u0()
                score, cm = do_run(u0)
                if score > 0.0
                    found_u0 = true
                    break
                end
                push!(traj, (copy(u0), score, cm))
            end
            if !found_u0
                @warn "Could not find a satisfactory u0 during prescreening"
            end
        else
            u0 = gen_u0()
        end

        # NOMAD run6
        @info "Starting NOMAD run $i"
        flush(stdout)
        function nomad_func(u)
            score, cm = do_run(u0)

            push!(traj, (copy(u), score, cm))

            (true, true, -score) # - as we want to maximize not minimize
        end

        numparams = length(u0)

        max1 = 10.0
        max3 = 100.0

        np = NOMAD.NomadProblem(numparams, 1, ["OBJ"], nomad_func;
            lower_bound=[
                -max1,
                0.0, 0.0,
                -max1,
                0.0, 0.0
            ],
            upper_bound=[
                max1,
                max3, max3,
                max1,
                1.0, 1.0
            ],
            # granularity=fill(granularity, numparams)
        )
        np.options.max_time = max_nomad_time
        np.options.display_all_eval = true
        np.options.display_stats = ["OBJ", "SOL", "BBE", "BBO"]

        GC.gc()
        @time s = NOMAD.solve(np, u0)

        push!(trajectories, traj)
        @info "Finished run $i"
        flush(stdout)
    end

    save_object("N$(N)_unic_" * randname() * ".jld2", trajectories)
    trajectories
end

function main3(;
    N=10,
    num_starts=30,
    num_prescreens=50,
    num_repeats=1000,
    max_nomad_time=60 * 60 * 0.1,
    max_single_solver_time=20,
    granularity=1e-3,
)
    BLAS.set_num_threads(1)

    function do_run(u0)
        me, la, lb, ce, sr, sb = u0
        do_run_unimodal_c(N,
            me, 0.0,
            -12.0, 0.0, 0.0, 0.0,
            la, lb,
            ce, 0.0,
            sr, sb,
            num_repeats,
            timelimit=max_single_solver_time,
        )
    end

    function gen_u0()
        [
            rand(Uniform(-2.0, 2.0)), # m
            1.5 * rand(), 1.5 * rand(), # l
            rand(Uniform(-1.0, 2.0)), # c
            rand(), rand(), # sparsities
        ]
    end

    trajectories = []

    for i in 1:num_starts
        traj = []

        # Prescreening
        @info "Prescreening $i"
        flush(stdout)
        if !isnothing(num_prescreens)
            found_u0 = false
            for _ in 1:num_prescreens
                u0 = gen_u0()
                score, cm = do_run(u0)
                if score > 0.0
                    found_u0 = true
                    break
                end
                push!(traj, (copy(u0), score, cm))
            end
            if !found_u0
                @warn "Could not find a satisfactory u0 during prescreening"
            end
        else
            u0 = gen_u0()
        end

        # NOMAD run6
        @info "Starting NOMAD run $i"
        flush(stdout)
        function nomad_func(u)
            score, cm = do_run(u0)

            push!(traj, (copy(u), score, cm))

            (true, true, -score) # - as we want to maximize not minimize
        end

        numparams = length(u0)

        max1 = 10.0
        max3 = 100.0

        np = NOMAD.NomadProblem(numparams, 1, ["OBJ"], nomad_func;
            lower_bound=[
                -max1,
                0.0, 0.0,
                -max1,
                0.0, 0.0
            ],
            upper_bound=[
                max1,
                max3, max3,
                max1,
                1.0, 1.0
            ],
            granularity=fill(granularity, numparams)
        )
        np.options.max_time = max_nomad_time
        np.options.display_all_eval = true
        np.options.display_stats = ["OBJ", "SOL", "BBE", "BBO"]

        GC.gc()
        @time s = NOMAD.solve(np, u0)

        push!(trajectories, traj)
        @info "Finished run $i"
        flush(stdout)
    end

    save_object("N$(N)_unic_" * randname() * ".jld2", trajectories)
    trajectories
end

################################################################################
# Plotting/analysis of the data
################################################################################
function plot_trajectory(traj;
    pnames=nothing,
    add_colorbar=true,
    vs_coloring=:time,
    tfilter=nothing,
    kwargs...
)
    if !isnothing(tfilter)
        traj = filter(tfilter, traj)
    end

    prep = prep_traj_plot(length(traj[1][1]), pnames)
    faa = plot_trajectory!(prep, traj; vs_coloring, kwargs...)

    if add_colorbar
        Colorbar(faa.figure[:, prep.num_ps+1], faa.obj.vs_plots[2, 1];
            label="vs plots - " * string(vs_coloring)
        )
    end

    faa
end

function plot_all_trajectories_obj2(ts;
    pnames=nothing,
    add_colorbar=true,
    vs_coloring=:time,
    tfilter=nothing,
    kwargs...
)
    newts = []
    if !isnothing(tfilter)
        for i in 1:length(ts)
            xx = filter(tfilter, ts[i])
            if !isempty(xx)
                push!(newts, xx)
            end
        end
    end
    ts = newts

    omin = Inf
    omax = -Inf
    for t in newts
        for x in t
            v = x[2]
            if v > omax
                omax = v
            end
            if v < omin
                omin = v
            end
        end
    end

    prep = prep_traj_plot(length(ts[1][1][1]), pnames)

    faas = []
    for t in ts
        push!(faas, plot_trajectory!(prep, t; vs_coloring, kwargs...))
    end

    if add_colorbar
        Colorbar(faas[1].figure[:, prep.num_ps+1], faas[1].obj.vs_plots[2, 1];
            label="vs plots - " * string(vs_coloring)
        )
    end

    faas[1]
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
                (1:length(traj)) ./ length(traj)
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
            ys = (1:length(traj)) ./ length(traj)
            ylabel = "time"
        elseif single_y == :obj
            ys = getindex.(traj, 2)
            ylabel = "objective"
        end

        sc_kwargs = Dict()
        if single_coloring == :time
            sc_kwargs[:color] = (1:length(traj)) ./ length(traj)
            clabel = "time"
        elseif single_coloring == :obj
            sc_kwargs[:color] = getindex.(traj, 2)
            clabel = "objective"
        elseif isnothing(single_coloring)
            clabel = nothing
        end

        sc = scatter!(ax, xs, ys; sc_kwargs..., single_kwargs...)
        push!(single_plots, sc)

        ax.ylabel = ylabel
        if !isnothing(clabel) && !occursin("color ~ ", ax.title[])
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
    fig = Figure(size=(2000, 2000 / 1.618))

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

function print_best_sols(ts)
    for t in ts
        _, i = findmax(x -> x[2], t)
        xx = t[i]
        @printf "obj: %10.5g <-> " xx[2]
        for (j, v) in enumerate(xx[1])
            @printf "%10.3g" v
        end
        println()
    end
end

function find_best_sols(ts;
    pnames=nothing
)
    if isnothing(pnames)
        numps = length(ts[1][1][1])
        pnames = ["p$i" for i in 1:numps]
    end

    df = DataFrame(;
        obj=Float64[],
        [Symbol(pn) => Float64[] for pn in pnames]...
    )
    for t in ts
        _, i = findmax(x -> x[2], t)
        xx = t[i]
        push!(df, (xx[2], xx[1]...))
    end

    df
end
