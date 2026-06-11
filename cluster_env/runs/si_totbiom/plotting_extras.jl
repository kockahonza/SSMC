include("../../../scripts/single_influx.jl")

using CairoMakie


function make_v1_report_group(f, group_i)
    fmetadata = f["metadata"]
    Klips = fmetadata.Klips_to_run

    N = fmetadata.N
    M = fmetadata.M
    dx = fmetadata.L / fmetadata.sN

    fode_rcs = f["ode_retcodes"]
    fsp_rcs = f["sp_retcodes"]

    fode_fss = f["ode_final_states"]
    fsp_fss = f["sp_final_states"]

    num_runs = fmetadata.num_runs

    fig = Figure(;
        size=(1200, 300 * num_runs)
    )

    Label(fig[1,1:3], (@sprintf "K=%.3g, l_influx=%.3g, p=%.3g" Klips[group_i]...), fontsize=25; tellwidth=false)

    Label(fig[2,1], "PDE final states", fontsize=20; tellwidth=false)
    Label(fig[2,2], "Comparing ODE vs PDE total abundances", fontsize=20; tellwidth=false)
    Label(fig[2,3], "Community composition across space", fontsize=20; tellwidth=false)

    spatial_xs = get_spatial_gridpoints_dx(fmetadata.sN, dx)

    for j in 1:num_runs
        ps = fparams[j, group_i]
        ode_fs = fode_fss[j, group_i]
        sp_fs = fsp_fss[j, group_i];

        ode_total_abundance = sum(ode_fs[1:N])
        isextinct = ode_total_abundance < 1e-6

        Label(fig[2+j, 0], "Run $j, $(fode_rcs[j, group_i]), $(fsp_rcs[j, group_i]), $isextinct";
            # tellwidth=false,
            tellheight=false,
            rotation=pi/2
        )

        if isextinct
            Label(fig[2+j, 1:3], "Extinct, skipping plot"; tellwidth=false, tellheight=false)
            continue
        end

        axs, axr = plot_spatial_fs!(fig[2+j, 1], sp_fs, 20, fmetadata.sN, dx, ode_fs)
        axr.xlabel = "Space"
        axs.ylabel = "Strains"
        axr.ylabel = "Resources"

        sp_means = mean(sp_fs, dims=2)[:,1]

        ax = Axis(fig[2+j, 2]; xlabel="ODE mean abundances", ylabel="PDE mean abundances")
        scatter!(ax, ode_fs[1:N], sp_means[1:N])

        rel_abundances = Matrix{Float64}(undef, N, fmetadata.sN)
        for i in axes(sp_fs, 2)
            strains_only = sp_fs[1:N, i]
            ss = sum(strains_only)
            rel_abundances[1:N, i] .= strains_only ./ ss
        end

        ode_rel_abundances = ode_fs[1:N] ./ ode_total_abundance;

        ax2 = Axis(fig[2+j, 3];
            xlabel="Space",
            ylabel="Relative abundance in space,\nODE levels dashed"
        )

        for i in 1:N
            lines!(ax2, spatial_xs, rel_abundances[i, :]; color=Cycled(i))
            hlines!(ax2, ode_rel_abundances[i]; linestyle=:dash, color=Cycled(i))
        end

        # plot_spatial_fs!(fig[2+j,3], rel_abundances, 20, fmetadata.sN, dx, ode_rel_abundances)
    end

    fig
end

function make_v1_report(f, out_dir="./")
    @show countmap(f["ode_retcodes"])
    @show countmap(f["sp_retcodes"]);

    fmetadata = f["metadata"]
    Klips = fmetadata.Klips_to_run

    num_groups = length(Klips)

    mkdir(out_dir)

    for i in 1:num_groups
        fig = make_v1_report_group(f, i)
        out_fname = joinpath(out_dir, "report_g$i.pdf")
        Makie.save(out_fname, fig)
    end
end
