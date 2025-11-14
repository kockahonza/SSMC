include("./base.jl")
include("../../../scripts/mm_Kl_pds.jl")

using Peaks

################################################################################
# Running the PDEs/ODEs
################################################################################
function do_v3_pde_run(
    logKs, ls, T,
    m, c,
    DN, DI, DR,
    N0, epsilon,
    L, sN;
    outfname="v3_run_" * timestamp() * ".jld2",
    solver=QNDF,
    tol=10000 * eps(),
    maxtime=30 * 60,
    kwargs...
)
    dx = L / sN
    u0 = clamp.(reduce(hcat, [[N0, 0.0, 0.0] .+ epsilon .* randn(3) for _ in 1:sN]), 0.0, Inf)

    params = Matrix{Any}(undef, length(logKs), length(ls))
    retcodes = Matrix{ReturnCode.T}(undef, length(logKs), length(ls))
    final_states = Matrix{Matrix{Float64}}(undef, length(logKs), length(ls))
    final_Ts = Matrix{Float64}(undef, length(logKs), length(ls))

    prog = Progress(length(logKs) * length(ls))
    @tasks for i in 1:length(logKs)
        logK = logKs[i]
        for (j, l) in enumerate(ls)
            mmp = MMParams(;
                K=10^logK,
                m,
                c,
                l=l,
                k=0.0,
                d=1.0,
            )
            sps = SASMMiCRMParams(
                mmp_to_mmicrm(mmp),
                SA[DN, DI, DR],
                make_cartesianspace_smart(1; dx),
                # nthreads()
            )
            sp = make_smmicrm_problem(sps, copy(u0), T)

            s = solve(sp, solver();
                dense=false,
                save_everystep=false,
                abstol=tol,
                reltol=tol,
                callback=make_timer_callback(maxtime),
                kwargs...
            )

            params[i, j] = sps
            retcodes[i, j] = s.retcode
            final_states[i, j] = s.u[end]
            final_Ts[i, j] = s.t[end]

            next!(prog)
            flush(stdout)
        end
    end
    finish!(prog)

    jldsave(outfname;
        logKs, ls, T,
        m, c,
        DN, DI, DR,
        N0, epsilon,
        L, sN,
        solver, tol, maxtime,
        params, retcodes, final_states, final_Ts,
    )

    (; params, retcodes, final_states, final_Ts)
end

function add_v3_nospace_run!(fname;
    solver=QNDF,
    tol=nothing,
    maxtime=5 * 60,
    kwargs...
)
    f = jldopen(fname, "r+")

    T = f["T"]
    N0 = f["N0"]
    u0 = [N0, 0.0, 0.0]

    tol_ = something(tol, f["tol"])

    params = f["params"]

    retcodes = Matrix{ReturnCode.T}(undef, size(params)...)
    final_states = Matrix{Vector{Float64}}(undef, size(params)...)
    final_Ts = Matrix{Float64}(undef, size(params)...)

    prog = Progress(length(params))
    @tasks for ci in eachindex(params)
        ps = params[ci].mmicrm_params
        p = make_mmicrm_problem(ps, copy(u0), T)

        s = solve(p, solver();
            abstol=tol_,
            reltol=tol_,
            callback=make_timer_callback(maxtime),
            kwargs...
        )

        retcodes[ci] = s.retcode
        final_states[ci] = s.u[end]
        final_Ts[ci] = s.t[end]
    end
    finish!(prog)

    f["ns_retcodes"] = retcodes
    f["ns_final_states"] = final_states
    f["ns_final_Ts"] = final_Ts

    close(f)

    (; retcodes, final_states, final_Ts)
end

################################################################################
# Specifying the runs we want to do
################################################################################
function v3main_base()
    outfname = "v3_base1.jld2"
    pde_results = do_v3_pde_run(
        range(-0.1, 3, 100),
        LeakageScale.lxrange(0.01, 0.99, 30),
        1e6,
        1.0, 1.0,
        1e-6, 1.0, 1.0,
        100.0, 1e-5,
        5, 5000;
        outfname,
    )
    @show countmap(pde_results.retcodes)
    nospace_results = add_v3_nospace_run!(outfname)
    @show countmap(nospace_results.retcodes)
end

function v3main_lowDR()
    outfname = "v3_lowDR1.jld2"
    pde_results = do_v3_pde_run(
        range(-0.1, 3, 100),
        LeakageScale.lxrange(0.01, 0.99, 30),
        1e6,
        1.0, 1.0,
        1e-6, 1.0, 0.01,
        100.0, 1e-5,
        5, 5000;
        outfname,
    )
    @show countmap(pde_results.retcodes)
    nospace_results = add_v3_nospace_run!(outfname)
    @show countmap(nospace_results.retcodes)
end

function v3main_lowm_lowDR()
    outfname = "v3_lowm_lowDR1.jld2"
    pde_results = do_v3_pde_run(
        range(-2.1, 0.5, 100),
        LeakageScale.lxrange(0.01, 0.99, 30),
        1e6,
        0.01, 1.0,
        1e-6, 1.0, 0.01,
        100.0, 1e-5,
        5, 5000;
        outfname,
    )
    @show countmap(pde_results.retcodes)
    nospace_results = add_v3_nospace_run!(outfname)
    @show countmap(nospace_results.retcodes)
end

function v3main1()
    v3main_base()
    v3main_lowDR()
    v3main_lowm_lowDR()
end

################################################################################
# Plotting/making consise reports
################################################################################
function make_quality_summary!(place, f)
    display(countmap(f["retcodes"]))
    display(countmap(f["ns_retcodes"]))

    logKs = f["logKs"]
    ls = f["ls"]
    leak_xs = LeakageScale.ltox.(ls)

    ax1 = MinimalModelV2.make_mm_Kl_hm_ax(place[1, 1], logKs, ls;
        title="Reached final time"
    )
    hm1 = heatmap!(ax1, 10 .^ logKs, leak_xs, f["final_Ts"];
        colorscale=log10
    )
    Colorbar(place[1, 2], hm1)

    maxresids = map(zip(f["params"], f["final_states"])) do (ps, fs)
        maximum(abs, uninplace(smmicrmfunc!)(fs, ps))
    end
    ax2 = MinimalModelV2.make_mm_Kl_hm_ax(place[1, 3], logKs, ls;
        title=(@sprintf "Maximal absolute residuals (max is %.3g)" maximum(maxresids))
    )
    hm2 = heatmap!(ax2, 10 .^ logKs, leak_xs, clamp_for_log(maxresids);
        # colorscale=log10
        colorscale=Makie.Symlog10(1e-6)
    )
    Colorbar(place[1, 4], hm2)

    ax3 = MinimalModelV2.make_mm_Kl_hm_ax(place[2, 1], logKs, ls;
        title="No-space reached final time"
    )
    hm3 = heatmap!(ax3, 10 .^ logKs, leak_xs, f["ns_final_Ts"];
        colorscale=log10
    )
    Colorbar(place[2, 2], hm3)

    nsmaxresids = map(zip(f["params"], f["ns_final_states"])) do (ps, fs)
        maximum(abs, uninplace(mmicrmfunc!)(fs, ps))
    end
    ax4 = MinimalModelV2.make_mm_Kl_hm_ax(place[2, 3], logKs, ls;
        title=(@sprintf "No-space maximal absolute residuals (max is %.3g)" maximum(nsmaxresids))
    )
    hm4 = heatmap!(ax4, 10 .^ logKs, leak_xs, nsmaxresids;
        # colorscale=log10
        colorscale=Makie.Symlog10(1e-6)
    )
    Colorbar(place[2, 4], hm4)

    for ax in [ax1, ax2, ax3, ax4]
        MinimalModelV2.draw_fr_lines2!(ax, ls, f["m"], f["c"], f["DR"] / f["DI"])
    end

    place
end
function make_quality_summary(f)
    fig = Figure(;
        size=(1000, 800)
    )
    make_quality_summary!(fig, f)
    fig
end

function make_basic_stats_plot!(place, f)
    logKs = f["logKs"]
    ls = f["ls"]
    leak_xs = LeakageScale.ltox.(ls)

    fs_N = map(f["final_states"]) do fs
        fs[1, :]
    end

    ax1 = MinimalModelV2.make_mm_Kl_hm_ax(place[1, 1], logKs, ls;
        title="Mean of final state N(x)",
    )
    hm1 = heatmap!(ax1, 10 .^ logKs, leak_xs, clamp_for_log(mean.(fs_N));
        colorscale=log10
    )
    Colorbar(place[1, 2], hm1)
    ax2 = MinimalModelV2.make_mm_Kl_hm_ax(place[1, 3], logKs, ls;
        title="Standard deviation of final state N(x)",
    )
    hm2 = heatmap!(
        ax2,
        10 .^ logKs,
        leak_xs,
        clamp_for_log(std.(fs_N));
        # colorscale=log10
    )

    for ax in [ax1, ax2]
        MinimalModelV2.draw_fr_lines2!(ax, ls, f["m"], f["c"], f["DR"] / f["DI"])
    end

    Colorbar(place[1, 4], hm2)
    place
end
function make_basic_stats_plot(f)
    fig = Figure(;
        size=(1000, 400)
    )
    make_basic_stats_plot!(fig, f)
    fig
end

function make_avgN_comparison_plot!(place, f)
    logKs = f["logKs"]
    ls = f["ls"]
    leak_xs = LeakageScale.ltox.(ls)

    wsNs = clamp_for_log(map(f["final_states"]) do fs
        mean(fs[1, :])
    end
    )
    nsNs = clamp_for_log(getindex.(f["ns_final_states"], 1))
    dNs = wsNs .- nsNs
    ext = extrema(vcat(wsNs, nsNs))

    gl_sbs = GridLayout(place[1, 1:2])

    ax1 = MinimalModelV2.make_mm_Kl_hm_ax(gl_sbs[1, 1], logKs, ls; title="With space")
    ax2 = MinimalModelV2.make_mm_Kl_hm_ax(gl_sbs[1, 2], logKs, ls; title="Without space")

    hm1 = heatmap!(ax1, 10 .^ logKs, leak_xs, wsNs;
        colorscale=log10,
        colorrange=ext
    )
    hm2 = heatmap!(ax2, 10 .^ logKs, leak_xs, nsNs;
        colorscale=log10,
        colorrange=ext
    )

    Colorbar(gl_sbs[1, 3], hm1)

    gl_dN = GridLayout(place[2, 1])

    ax3 = MinimalModelV2.make_mm_Kl_hm_ax(gl_dN[1, 1], logKs, ls; title="Difference (with - without)")
    hm3 = heatmap!(ax3, 10 .^ logKs, leak_xs, dNs;
    )
    Colorbar(gl_dN[1, 2], hm3)

    gl_dT = GridLayout(place[2, 2])
    ax4 = MinimalModelV2.make_mm_Kl_hm_ax(gl_dT[1, 1], logKs, ls; title="Final T Difference (with - without)")
    hm4 = heatmap!(ax4, 10 .^ logKs, leak_xs, f["final_Ts"] .- f["ns_final_Ts"];
    )
    Colorbar(gl_dT[1, 2], hm4)

    for ax in [ax1, ax2, ax3, ax4]
        MinimalModelV2.draw_fr_lines2!(ax, ls, f["m"], f["c"], f["DR"] / f["DI"])
    end

    place
end
function make_avgN_comparison_plot(f)
    fig = Figure(;
        size=(1000, 800)
    )
    make_avgN_comparison_plot!(fig, f)
    fig
end

function get_peaks(fss, dx)
    full_peaks = map(fss) do fs
        all_pks = findmaxima(fs[1, :])
        peakproms(all_pks; min=0.1)
    end
    numpeaks = map(full_peaks) do pks
        length(pks.indices)
    end
    avg_pkh = map(full_peaks) do pks
        mean(pks.heights)
    end
    avg_pkw = map(full_peaks) do pks
        if length(pks.indices) > 0
            mean(peakwidths(pks).widths) * dx
        else
            missing
        end
    end
    avg_pkp = map(full_peaks) do pks
        if length(pks.indices) > 0
            mean(peakproms(pks).proms)
        else
            missing
        end
    end
    avg_pksp = map(full_peaks) do pks
        if length(pks.indices) > 1
            mean(pks.indices[2:end] .- pks.indices[1:end-1]) * dx
        else
            missing
        end
    end
    (; full_peaks, numpeaks, avg_pkh, avg_pkw, avg_pkp, avg_pksp)
end
function get_peaks(f)
    get_peaks(f["final_states"], f["L"] / f["sN"])
end

function make_peaks_full_plot!(place, f)
    logKs = f["logKs"]
    ls = f["ls"]
    leak_xs = LeakageScale.ltox.(ls)

    (; full_peaks, numpeaks, avg_pkh, avg_pkw, avg_pkp, avg_pksp) = get_peaks(f)

    gl1 = GridLayout(place[1, 1])
    ax1 = MinimalModelV2.make_mm_Kl_hm_ax(gl1[1, 1], logKs, ls; title="Number of peaks")
    hm1 = heatmap!(
        ax1,
        10 .^ logKs,
        leak_xs,
        numpeaks;
        # colorscale=log10,
    )
    Colorbar(gl1[1, 2], hm1)

    gl2 = GridLayout(place[1, 2])
    ax2 = MinimalModelV2.make_mm_Kl_hm_ax(gl2[1, 1], logKs, ls; title="Avg peak prominence")
    hm2 = heatmap!(
        ax2,
        10 .^ logKs,
        leak_xs,
        avg_pkp;
        # colorscale=log10,
    )
    Colorbar(gl2[1, 2], hm2)

    gl3 = GridLayout(place[2, 1])
    ax3 = MinimalModelV2.make_mm_Kl_hm_ax(gl3[1, 1], logKs, ls; title="Avg peak width")
    hm3 = heatmap!(ax3, 10 .^ logKs, leak_xs, avg_pkw;
        colorscale=log10,
    )
    Colorbar(gl3[1, 2], hm3)

    gl4 = GridLayout(place[2, 2])
    ax4 = MinimalModelV2.make_mm_Kl_hm_ax(gl4[1, 1], logKs, ls; title="Avg peak spacing")
    hm4 = heatmap!(
        ax4,
        10 .^ logKs,
        leak_xs,
        avg_pksp;
        # colorscale=log10,
    )
    Colorbar(gl4[1, 2], hm4)

    place
end
function make_peaks_full_plot(f)
    fig = Figure(;
        size=(1000, 800)
    )
    make_peaks_full_plot!(fig, f)
    fig
end

function make_full_report_plot1(f::JLD2.JLDFile)
    fig = Figure(;
        size=(800, 2200)
    )

    Label(fig[1, 1], "Data quality summary";
        fontsize=30,
        tellwidth=false,
    )
    make_quality_summary!(fig[2, 1], f)

    Label(fig[3, 1], "Final state average N summary";
        fontsize=30,
        tellwidth=false,
    )
    make_avgN_comparison_plot!(fig[4, 1], f)

    Label(fig[5, 1], "Spatial final state mean+std of N accross space";
        fontsize=30,
        tellwidth=false,
    )
    make_basic_stats_plot!(fig[6, 1], f)

    Label(fig[7, 1], "Spatial final state N peaks stats";
        fontsize=30,
        tellwidth=false,
    )
    make_peaks_full_plot!(fig[8, 1], f)

    Label(fig[9, 1], "Analytic+Semi-analytic results";
        fontsize=30,
        tellwidth=false,
    )
    gl = GridLayout(fig[10, 1])
    make_Kl_pd!(gl[1, 1], f["logKs"], f["ls"], f["m"], f["c"], f["DN"], f["DI"], f["DR"])
    make_Kl_pd_legend_full!(gl[1, 2];
        labelsize=10,
        # orientation=:horizontal
    )

    rowsize!(fig.layout, 6, Relative(0.1))
    rowsize!(fig.layout, 10, Relative(0.1))

    fig
end
function make_full_report_plot1(fname::AbstractString, outfname=nothing)
    f = jldopen(fname)
    fig = make_full_report_plot1(f)
    close(f)

    Label(fig[0, 1], "Summary report for:\n$(basename(fname))";
        fontsize=50,
        tellwidth=false,
    )

    if outfname != false
        if isnothing(outfname)
            outfname = joinpath(dirname(fname), "frep_" * splitext(basename(fname))[1] * ".pdf")
        end
        @show outfname
        Makie.save(outfname, fig)
    end

    fig
end
