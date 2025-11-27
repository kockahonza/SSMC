using Revise

includet("../../../scripts/single_influx.jl")

import DataFrames: metadata as md

################################################################################
# Shared/general/util
################################################################################
function run_1d_pde_sim(ps, u0, T, L, sN;
    epsilon=1e-3,
    # solver options
    tol=100000 * eps(),
    maxtime=5 * 60 * 60,
    solver=QNDF,
    solver_threads=nthreads(),
)
    N, M = get_Ns(ps)
    dx = L / sN

    u0_ = if ndims(u0) == 0
        expand_u0_to_size((sN,), [fill(u0, N); fill(0.0, M)])
    elseif ndims(u0) == 1
        expand_u0_to_size((sN,), u0)
    elseif ndims(u0) == 2
        copy(u0)
    else
        throw(ArgumentError("invalid u0 passed to run_1d_pdes_from_df"))
    end
    if !isnothing(epsilon)
        u0_ = perturb_u0_uniform(N, M, u0_, epsilon)
    end

    sps = BSMMiCRMParams(
        ps.mmicrm_params,
        ps.Ds,
        CartesianSpace{1,Tuple{Periodic}}(SA[dx]),
        solver_threads
    )
    sp = make_smmicrm_problem(sps, u0_, T)

    s = solve(sp, solver();
        dense=false,
        save_everystep=false,
        calck=false,
        abstol=tol,
        reltol=tol,
        callback=make_timer_callback(maxtime)
    )

    s
end

################################################################################
# First set of functions - runs PDES near steady states
################################################################################
function run_1d_pdes_from_df(fname;
    T=1e6,
    L=5,              # system size in non-dim units
    sN=5000,          # number of spatial points
    epsilon=1e-3,     # perturbation amplitude
    # threading
    run_threads=1,
    total_threads=nthreads(),
    solver_threads=div(total_threads, run_threads),
    # solver options
    maxtime=5 * 60 * 60,
    kwargs...
)
    f = jldopen(fname)
    df = copy(f["df"])
    close(f)

    N = md(df, "N")
    M = md(df, "M")

    metadata!(df, "T", T; style=:note)
    metadata!(df, "L", L; style=:note)
    metadata!(df, "sN", sN; style=:note)
    metadata!(df, "epsilon", epsilon; style=:note)

    num_runs = nrow(df)

    retcodes = Vector{ReturnCode.T}(undef, num_runs)
    final_states = Vector{Matrix{Float64}}(undef, num_runs)
    final_Ts = Vector{Float64}(undef, num_runs)

    @printf "Starting a base run on %s which contains %d rows\n" fname num_runs
    flush(stdout)

    prog = Progress(num_runs)
    @tasks for i in 1:num_runs
        @set ntasks = run_threads
        ps = df.params[i]
        ss = df.steadystates[i]

        s = run_1d_pde_sim(ps, ss, T, L, sN;
            epsilon,
            maxtime,
            solver_threads,
            kwargs...
        )

        retcodes[i] = s.retcode
        final_states[i] = s.u[end]
        final_Ts[i] = s.t[end]

        s = nothing
        GC.gc()

        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)

    df.retcodes = retcodes
    df.final_states = final_states
    df.final_Ts = final_Ts

    df
end

"""
Takes a file with a df with params and hss for which it will run a spatial pde starting from a
parturbed ss. Adds the resulting final states and times to the df and saves into a new file
"""
function main1()
    df = run_1d_pdes_from_df("./sel_systems3.jld2";
        run_threads=2,
        solver_threads=64,
        maxtime=5 * 60 * 60,
    )
    jldsave("./rslt_df3.jld2"; df)
end

################################################################################
# Second set of functions - also adds high N0 pde runs to result runs from main1
################################################################################
function add_highN0_run(fname, outfname=nothing;
    prefix="highN0_",
    N0=100.0,
    # threading
    run_threads=1,
    total_threads=nthreads(),
    solver_threads=div(total_threads, run_threads),
    # solver options
    maxtime=5 * 60 * 60,
    kwargs...
)
    f = jldopen(fname)
    df = copy(f["df"])
    close(f)

    metadata!(df, "N0", N0; style=:note)

    T = md(df, "T")
    L = md(df, "L")
    sN = md(df, "sN")
    epsilon = md(df, "epsilon")

    num_runs = nrow(df)

    retcodes = Vector{ReturnCode.T}(undef, num_runs)
    final_states = Vector{Matrix{Float64}}(undef, num_runs)
    final_Ts = Vector{Float64}(undef, num_runs)

    @printf "Starting a high N0 run on %s which contains %d rows\n" fname num_runs
    flush(stdout)

    prog = Progress(num_runs)
    @tasks for i in 1:num_runs
        @set ntasks = run_threads
        ps = df.params[i]

        s = run_1d_pde_sim(ps, N0, T, L, sN;
            epsilon,
            maxtime,
            solver_threads,
            kwargs...
        )

        retcodes[i] = s.retcode
        final_states[i] = s.u[end]
        final_Ts[i] = s.t[end]

        s = nothing
        GC.gc()

        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)

    df[:, prefix*"retcodes"] = retcodes
    df[:, prefix*"final_states"] = final_states
    df[:, prefix*"final_Ts"] = final_Ts

    if !isnothing(outfname)
        jldsave(outfname; df)
    end

    df
end

"""
Adds a high N0 run to the same file as would be outputted by main1
"""
function main2()
    df = add_highN0_run("./rslt_df3.jld2", "./rslt2_df3.jld2";
        run_threads=2,
        solver_threads=64,
        maxtime=5 * 60 * 60,
    )
    jldsave("./forsafety.jld2"; df)
end

################################################################################
# Do both in the same cluster run
################################################################################
function main3()
    df = run_1d_pdes_from_df("./sel_systems4.jld2";
        run_threads=4,
        solver_threads=32,
        maxtime=5 * 60 * 60,
    )
    jldsave("./rslt_df4.jld2"; df)
    df = add_highN0_run("./rslt_df4.jld2", "./rslt2_df4.jld2";
        run_threads=4,
        solver_threads=32,
        maxtime=5 * 60 * 60,
    )
    jldsave("./forsafety.jld2"; df)
end

function main3_v2_fname(runname)
    input_fname = "./selsystems_v2_$(runname).jld2"
    out1_fname = "./rslt1_$(runname).jld2"
    out2_fname = "./rslt2_$(runname).jld2"
    df = run_1d_pdes_from_df(input_fname;
        run_threads=4,
        solver_threads=32,
        maxtime=5 * 60 * 60,
    )
    jldsave(out1_fname; df)
    df = add_highN0_run(out1_fname, out2_fname;
        run_threads=4,
        solver_threads=32,
        maxtime=5 * 60 * 60,
    )
end

################################################################################
# Plotting/fast analysis bits
################################################################################
using Makie

function add_mean_final_states(df)
    df.mfss = map(df.final_states) do fs
        mean(fs; dims=2)[:, 1]
    end
    df.highN0_mfss = map(df.highN0_final_states) do fs
        mean(fs; dims=2)[:, 1]
    end
end

function add_Ks(df)
    Ks = map(df.params) do ps
        iis = findall(!iszero, ps.K)
        if length(iis) != 1
            throw(ArgumentError("Encountered a system with multiple influx resources (or none at all)"))
        end
        ps.K[iis[1]]
    end
    df.Ks = Ks
end

function plot_spatial_fs!(gl, u, Ns, sN, dx, ss=nothing;
    axis=(;),
)
    Nr = size(u)[1] - Ns
    xs = ((1:sN) .- 0.5) .* dx

    axs = Axis(gl[1, 1]; axis...)
    axr = Axis(gl[2, 1]; axis...)
    linkxaxes!(axs, axr)
    hidexdecorations!(axs)
    rowgap!(gl, 4.0)

    for i in 1:Ns
        lines!(axs, xs, u[i, :]; color=Cycled(i))
    end
    for a in 1:Nr
        lines!(axr, xs, u[Ns+a, :]; color=Cycled(Ns + a))
    end

    if !isnothing(ss)
        for i in 1:Ns
            hlines!(axs, ss[i];
                color=Cycled(i),
                linestyle=:dash
            )
        end
        for a in 1:Nr
            hlines!(axr, ss[Ns+a];
                color=Cycled(Ns + a),
                linestyle=:dash
            )
        end
    end

    axs, axr
end
function plot_spatial_fs(args...;
    figure=(;),
    kwargs...
)
    fig = Figure(; figure...)
    plot_spatial_fs!(fig, args...; kwargs...)

    fig
end

function make_spatial_report(df;
    rowsize=200,
    width=1000,
)
    N = md(df, "N")
    M = md(df, "M")
    T = md(df, "T")
    L = md(df, "L")
    sN = md(df, "sN")
    dx = md(df, "dx", L / sN)

    N0 = md(df, "N0")

    num_runs = nrow(df)
    fig = Figure(;
        size=(width, rowsize * num_runs)
    )

    for ri in 1:num_runs
        sr = df[ri, :]

        bgl = GridLayout(fig[ri, 1])
        baxs, baxr = plot_spatial_fs!(bgl, sr.final_states, N, sN, dx, df.steadystates[ri])

        hN0gl = GridLayout(fig[ri, 2])
        hN0axs, hN0axr = plot_spatial_fs!(hN0gl, sr.highN0_final_states, N, sN, dx, df.steadystates[ri])

        # baxr.xlabel = "space"
        # hN0axr.xlabel = "space"

        Label(fig[ri, 0], "Params $ri";
            tellheight=false,
            rotation=pi / 2,
        )
    end
    Label(fig[0, 1], "Starting from near homogeneous steady state"; tellwidth=false)
    Label(fig[0, 2], "Starting from high N0=$(N0)"; tellwidth=false)

    rowgap!(fig.layout, 5.0)

    fig
end

# Scatterplot thing
function make_mfss_scatterplot(df)
    fig = Figure(;
        size=(400, 400),
    )
    ax = Axis(fig[1, 1];
        xlabel="No-space final state mean abundance",
        ylabel="With space final state mean abundance",
        aspect=DataAspect(),
    )

    mm = max(maximum(maximum, df.mfss), maximum(maximum, df.steadystates))
    lines!(ax, [0, mm], [0, mm]; color=:black)

    scatter!(ax,
        reduce(vcat, [x[1:N] for x in df.steadystates]),
        reduce(vcat, [x[1:N] for x in df.mfss]);
        label="Strains"
    )
    scatter!(ax,
        reduce(vcat, [x[(N+1):(N+M)] for x in df.steadystates]),
        reduce(vcat, [x[(N+1):(N+M)] for x in df.mfss]);
        label="Resources"
    )
    axislegend(ax)

    fig
end
function make_mfss_scatterplot2(df)
    fig = Figure(;
    # size=(800,800),
    )
    ax = Axis(fig[1, 1];
        xlabel="No-space final state mean abundance",
        ylabel="With space final state mean abundance",
        # aspect=DataAspect(),
    )
    scatter!(ax,
        reduce(vcat, [x[1:N] for x in df.steadystates]) .- reduce(vcat, [x[1:N] for x in df.mfss]);
        label="Strains"
    )
    scatter!(ax,
        reduce(vcat, [x[(N+1):(N+M)] for x in df.steadystates]) .- reduce(vcat, [x[(N+1):(N+M)] for x in df.mfss]);
        label="Resources"
    )
    axislegend(ax)

    fig
end

# function make_strains_mfss_scatterplot(df)
# end
