function plot_smmicrm_sol_avgs(sol, is=1:length(sol.u); singleax=false, plote=false)
    params = sol.prob.p
    if !isa(params, SMMiCRMParams)
        throw(ArgumentError("this func can only plot solutions of SMMiCRM problems"))
    end
    Ns, Nr = get_Ns(params.mmicrm_params)

    ts = sol.t[is]
    avgs = [mean(u, dims=2:ndims(u)) for u in sol.u[is]]
    energies = calc_E.(avgs, Ref(sol.prob.p.mmicrm_params))

    fig = Figure()
    if singleax
        strainax = resax = Axis(fig[1, 1])
        if plote
            eax = strainax
        end
    else
        strainax = Axis(fig[1, 1])
        resax = Axis(fig[2, 1])
        if plote
            eax = Axis(fig[3, 1])
        end

        # Link x axes
        linkxaxes!(strainax, resax)
        if plote
            linkxaxes!(strainax, eax)
        end
    end

    # plot data
    for i in 1:Ns
        scatterlines!(strainax, ts, getindex.(avgs, i); label=@sprintf "str %d" i)
    end
    for a in 1:Nr
        scatterlines!(resax, ts, getindex.(avgs, Ns + a); label=@sprintf "res %d" a)
    end
    if plote
        scatterlines!(eax, ts, energies; label=L"\epsilon")
    end

    if singleax
        axislegend(strainax)
    else
        axislegend(strainax)
        axislegend(resax)
        if plote
            axislegend(eax)
        end
    end
    fig
end
export plot_smmicrm_sol_avgs

################################################################################
# 1D
################################################################################
function plot_1dsmmicrm_sol_snap(params, snap_u; singleax=false, plote=false)
    if !isa(params, SMMiCRMParams) || (ndims(params) != 1)
        throw(ArgumentError("plot_smmicrm_sol_snap can only plot 1D solutions of SMMiCRM problems"))
    end
    Ns, Nr = get_Ns(params.mmicrm_params)

    len = size(snap_u)[2]
    xs = params.space.dx[1] .* (0:(len-1))

    fig = Figure()
    if singleax
        strainax = resax = Axis(fig[1, 1])
        if plote
            eax = strainax
        end
    else
        strainax = Axis(fig[1, 1])
        resax = Axis(fig[2, 1])
        if plote
            eax = Axis(fig[3, 1])
        end
    end

    # plot data
    for i in 1:Ns
        lines!(strainax, xs, snap_u[i, :]; label=@sprintf "str %d" i)
    end
    for a in 1:Nr
        lines!(resax, xs, snap_u[Ns+a, :]; label=@sprintf "res %d" a)
    end
    # if plote
    #     lines!(eax, xs, calc_E.(sol.u, Ref(params)); label=L"\epsilon")
    # end

    if singleax
        axislegend(strainax)
    else
        axislegend(strainax)
        axislegend(resax)
        if plote
            axislegend(eax)
        end
    end

    if singleax
        axs = strainax
    else
        axs = [strainax, resax]
    end

    FigureAxisAnything(fig, axs, nothing)
end
export plot_1dsmmicrm_sol_snap

################################################################################
# 2D
################################################################################

"""
Base function for creating 2D heatmap plots of SMMiCRM solutions.
Returns (fig, strain_hms, resource_hms, time_label) where the heatmaps can be updated.
"""
function setup_2dsmmicrm_heatmap_figure(params::SMMiCRMParams, x_range, y_range;
    size=nothing, aspect_ratio=1.5, time_value=nothing,
    strain_clims=nothing, resource_clims=nothing)

    Ns, Nr = get_Ns(params.mmicrm_params)

    # Create figure with optional size
    fig = isnothing(size) ? Figure() : Figure(; size=size)

    # Calculate grid layout for strains and resources
    strain_rows, strain_cols = make_grid(Ns; aspect_ratio)
    resource_rows, resource_cols = make_grid(Nr; aspect_ratio)

    # Create layout for strain heatmaps and resource heatmaps
    strain_panel = fig[1, 1] = GridLayout()
    resource_panel = fig[2, 1] = GridLayout()
    colorbar_panel = fig[1:2, 2] = GridLayout()

    # Add time label if requested
    time_label = nothing
    if !isnothing(time_value)
        if isa(time_value, Number)
            time_label = Label(fig[3, 1], @sprintf "t = %.2f" time_value)
        elseif isa(time_value, String)
            time_label = Label(fig[3, 1], @sprintf "t = %s" time_value)
        else
            time_label = Label(fig[3, 1], "")
        end
    end

    # Create axes for strains and resources using the calculated grid
    strain_axes = Matrix{Union{Axis,Nothing}}(nothing, strain_rows, strain_cols)
    resource_axes = Matrix{Union{Axis,Nothing}}(nothing, resource_rows, resource_cols)

    # Create heatmaps for each strain
    strain_hms = []
    for i in 1:Ns
        row = div(i - 1, strain_cols) + 1
        col = mod(i - 1, strain_cols) + 1
        ax = Axis(strain_panel[row, col]; aspect=DataAspect())
        strain_axes[row, col] = ax

        hm = heatmap!(ax, x_range, y_range, zeros(length(x_range), length(y_range));  # placeholder data
            colorrange=strain_clims, colormap=:viridis)
        push!(strain_hms, hm)
        ax.title = "Strain $i"
        ax.xlabel = "x"
        ax.ylabel = "y"
    end

    # Create heatmaps for each resource
    resource_hms = []
    for i in 1:Nr
        row = div(i - 1, resource_cols) + 1
        col = mod(i - 1, resource_cols) + 1
        ax = Axis(resource_panel[row, col]; aspect=DataAspect())
        resource_axes[row, col] = ax

        hm = heatmap!(ax, x_range, y_range, zeros(length(x_range), length(y_range));  # placeholder data
            colorrange=resource_clims, colormap=:plasma)
        push!(resource_hms, hm)
        ax.title = "Resource $i"
        ax.xlabel = "x"
        ax.ylabel = "y"
    end

    # Link axes
    for sax in strain_axes[2:end]
        if !isnothing(sax)
            linkxaxes!(strain_axes[1], sax)
            linkyaxes!(strain_axes[1], sax)
        end
    end
    for rax in resource_axes[:]
        if !isnothing(rax)
            linkxaxes!(strain_axes[1], rax)
            linkyaxes!(strain_axes[1], rax)
        end
    end

    # Add colorbars
    Colorbar(colorbar_panel[1, 1], strain_hms[1], label="Strain concentration")
    Colorbar(colorbar_panel[2, 1], resource_hms[1], label="Resource concentration")

    # fix the layout
    colsize!(fig.layout, 1, Auto(false))

    return (fig, strain_hms, resource_hms, time_label)
end

"""
Updates the heatmaps with new data.
"""
function update_2dsmmicrm_heatmaps!(strain_hms, resource_hms, u, Ns)
    # Update strain heatmaps
    for (i, hm) in enumerate(strain_hms)
        hm[3][] = permutedims(u[i, :, :])
    end

    # Update resource heatmaps
    for (i, hm) in enumerate(resource_hms)
        hm[3][] = permutedims(u[Ns+i, :, :])
    end
end

function plot_2dsmmicrm_sol_snap_heatmap(params::SMMiCRMParams, u, t=nothing; aspect_ratio=1.5)
    Ns = get_Ns(params.mmicrm_params)[1]

    # Create colormap ranges for better visualization
    strain_clims = (minimum(u[1:Ns, :, :]), maximum(u[1:Ns, :, :]))
    resource_clims = (minimum(u[Ns+1:end, :, :]), maximum(u[Ns+1:end, :, :]))

    # Setup the figure
    space_ranges = get_u_axes(u, params.space.dx)
    fig, strain_hms, resource_hms, _ = setup_2dsmmicrm_heatmap_figure(
        params, space_ranges[1], space_ranges[2];
        aspect_ratio=aspect_ratio,
        time_value=t,
        strain_clims=strain_clims,
        resource_clims=resource_clims
    )

    # Update with data
    update_2dsmmicrm_heatmaps!(strain_hms, resource_hms, u, Ns)

    fig
end
function plot_2dsmmicrm_sol_snap_heatmap(sol, t; kwargs...)
    if isa(t, Integer)
        if t < 0
            t = length(sol.t) + t + 1
        end
        u = sol.u[t]
        t = sol.t[t]
    else
        u = sol(t)
    end
    plot_2dsmmicrm_sol_snap_heatmap(sol.prob.p, u, t; kwargs...)
end
export plot_2dsmmicrm_sol_snap_heatmap

function plot_2dsmmicrm_sol_interactive_heatmap(sol; aspect_ratio=1.5)
    params = sol.prob.p
    if !isa(params, SMMiCRMParams)
        throw(ArgumentError("this func can only plot solutions of SMMiCRM problems"))
    end
    Ns = get_Ns(params.mmicrm_params)[1]

    # Create colormap ranges for better visualization
    strain_clims = (minimum(minimum(u[1:Ns, :, :]) for u in sol.u),
        maximum(maximum(u[1:Ns, :, :]) for u in sol.u))
    resource_clims = (minimum(minimum(u[Ns+1:end, :, :]) for u in sol.u),
        maximum(maximum(u[Ns+1:end, :, :]) for u in sol.u))

    # Setup the figure
    space_ranges = get_u_axes(sol.u[1], params.space.dx)
    fig, strain_hms, resource_hms, _ = setup_2dsmmicrm_heatmap_figure(
        params, space_ranges[1], space_ranges[2];
        aspect_ratio=aspect_ratio,
        strain_clims=strain_clims,
        resource_clims=resource_clims
    )

    # Add slider
    slider_layout = fig[3, 1] = GridLayout()
    timesl = Slider(slider_layout[1, 1], range=1:length(sol.t), startvalue=1)
    time_label = Label(slider_layout[1, 2], @lift(string("t = ", round(sol.t[$(timesl.value)], digits=2))))

    # Create on value change handler for slider
    on(timesl.value) do idx
        # Update strain heatmaps
        for (i, hm) in enumerate(strain_hms)
            hm[3][] = permutedims(sol.u[idx][i, :, :])
        end

        # Update resource heatmaps
        for (i, hm) in enumerate(resource_hms)
            hm[3][] = permutedims(sol.u[idx][Ns+i, :, :])
        end
    end

    # Initialize with first frame
    for (i, hm) in enumerate(strain_hms)
        hm[3][] = permutedims(sol.u[1][i, :, :])
    end

    for (i, hm) in enumerate(resource_hms)
        hm[3][] = permutedims(sol.u[1][Ns+i, :, :])
    end


    fig
end
export plot_2dsmmicrm_sol_interactive_heatmap

function plot_2dsmmicrm_sol_animation_heatmap(sol, filename=datadir(randname() * ".mp4"); size=(600, 400), fps=30, duration=10, aspect_ratio=1.5)
    params = sol.prob.p
    if !isa(params, SMMiCRMParams)
        throw(ArgumentError("this func can only plot solutions of SMMiCRM problems"))
    end
    Ns = get_Ns(params.mmicrm_params)[1]

    # Calculate total frames based on fps and duration
    total_frames = fps * duration
    time_indices = round.(Int, range(1, length(sol.t), length=total_frames))


    # Create colormap ranges for better visualization
    strain_clims = (minimum(minimum(u[1:Ns, :, :]) for u in sol.u),
        maximum(maximum(u[1:Ns, :, :]) for u in sol.u))
    resource_clims = (minimum(minimum(u[Ns+1:end, :, :]) for u in sol.u),
        maximum(maximum(u[Ns+1:end, :, :]) for u in sol.u))

    # Setup the figure
    space_ranges = get_u_axes(sol.u[1], params.space.dx)
    fig, strain_hms, resource_hms, time_label = setup_2dsmmicrm_heatmap_figure(
        params, space_ranges[1], space_ranges[2];
        size=size,
        aspect_ratio=aspect_ratio,
        time_value=@sprintf("t = %.2f", sol.t[1]),
        strain_clims=strain_clims,
        resource_clims=resource_clims
    )

    # Initial data
    update_2dsmmicrm_heatmaps!(strain_hms, resource_hms, sol.u[1], Ns)

    # Create animation
    framerate = fps
    record(fig, filename, time_indices; framerate=framerate) do frame_idx
        # Update time label
        time_label.text = @sprintf("t = %.2f", sol.t[frame_idx])

        # Update heatmaps
        update_2dsmmicrm_heatmaps!(strain_hms, resource_hms, sol.u[frame_idx], Ns)
    end

    fig
end
export plot_2dsmmicrm_sol_animation_heatmap

################################################################################
# Specific plots which are not general but still useful!
################################################################################
