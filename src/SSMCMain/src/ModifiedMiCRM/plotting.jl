function plot_mmicrm_sol(sol;
    singleax=false,
    plote=false,
    legends=length(sol.u[end]) < 15 ? true : false
)
    params = sol.prob.p
    if !isa(params, AbstractMMiCRMParams)
        throw(ArgumentError("plot_mmicrm_sol can only plot solutions of MMiCRM problems"))
    end
    Ns, Nr = get_Ns(params)

    fig = Figure()
    if singleax
        strainax = resax = Axis(fig[1, 1])
        if plote
            eax = strainax
        end
    else
        strainax = Axis(fig[1, 1])
        resax = Axis(fig[2, 1])
        linkxaxes!(strainax, resax)
        if plote
            eax = Axis(fig[3, 1])
            linkxaxes!(strainax, eax)
        end
    end

    # plot data
    for i in 1:Ns
        scatterlines!(strainax, sol.t, sol[i, :];
            label=(@sprintf "str %d" i),
            marker=:vline,
        )
    end
    for a in 1:Nr
        scatterlines!(resax, sol.t, sol[Ns+a, :];
            label=(@sprintf "res %d" a),
            marker=:vline,
        )
    end
    if plote
        scatterlines!(eax, sol.t, calc_E.(sol.u, Ref(params));
            label=L"\epsilon",
            marker=:vline,
        )
    end

    if legends
        if singleax
            axislegend(strainax)
        else
            axislegend(strainax)
            axislegend(resax)
            if plote
                axislegend(eax)
            end
        end
    end
    fig
end
export plot_mmicrm_sol

function plot_linstab_lambdas(ks, lambdas; imthreshold=1e-8)
    fig = Figure()
    ax = Axis(fig[1, 1])

    num_lambdas = length(lambdas[1])

    for li in 1:num_lambdas
        ls = [lambdas[i][li] for i in 1:length(lambdas)]

        lines!(ax, ks, real(ls);
            color=Cycled(li),
            label=latexstring(@sprintf "\\Re(\\lambda_%d)" li)
        )
        ims = imag(ls)

        mims = maximum(abs, ims)
        if mims > imthreshold
            # @info @sprintf "we are getting non-zero imaginary parts, max(abs(.)) is %f" mims
            lines!(ax, ks, ims;
                color=Cycled(li),
                linestyle=:dash,
                label=latexstring(@sprintf "\\Im(\\lambda_%d)" li)
            )
        end
    end
    axislegend(ax)

    mrl = maximum(ls -> maximum(real, ls), lambdas)
    if mrl > 1000 * eps()
        @info @sprintf "Unstable, mrl is %g" mrl
        ylims!(ax, -0.2 * abs(mrl), 1.5 * abs(mrl))
    end

    FigureAxisAnything(fig, ax, lambdas)
end
export plot_linstab_lambdas

################################################################################
# Spatial
################################################################################
function plot_smmicrm_sol_avgs(sol, is=:; singleax=false, plote=false)
    params = sol.prob.p
    if !isa(params, AbstractSMMiCRMParams)
        throw(ArgumentError("this func can only plot solutions of SMMiCRM problems"))
    end
    Ns, Nr = get_Ns(params.mmicrm_params)

    if isa(is, Colon)
        is = 1:length(sol.u)
    end

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
        scatterlines!(strainax, ts, getindex.(avgs, i);
            marker=:vline,
            label=@sprintf "str %d" i
        )
    end
    for a in 1:Nr
        scatterlines!(resax, ts, getindex.(avgs, Ns + a);
            marker=:vline,
            label=@sprintf "res %d" a
        )
    end
    if plote
        scatterlines!(eax, ts, energies;
            marker=:vline,
            label=L"\epsilon"
        )
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

# 1D
"""
Base function for setting up 1D SMMiCRM solution plots.
Returns (fig, strain_lines, resource_lines, time_label) where the lines can be updated.
"""
function setup_1dsmmicrm_figure(params::AbstractSMMiCRMParams, xs;
    singleax=false, plote=false, time_value=nothing
)
    if ndims(params) != 1
        throw(ArgumentError("This function can only plot 1D solutions of SMMiCRM problems"))
    end
    Ns, Nr = get_Ns(params.mmicrm_params)


    fig = Figure()

    # Create axes based on singleax parameter
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
        linkxaxes!(strainax, resax)
        if plote
            linkxaxes!(strainax, eax)
        end
    end

    # Initialize empty lines for strains and resources
    strain_lines = [lines!(strainax, xs, zeros(length(xs)); label=@sprintf("str %d", i)) for i in 1:Ns]
    resource_lines = [lines!(resax, xs, zeros(length(xs)); label=@sprintf("res %d", i)) for i in 1:Nr]

    # Add legends
    if singleax
        axislegend(strainax)
    else
        axislegend(strainax)
        axislegend(resax)
        if plote
            axislegend(eax)
        end
    end

    # Add time label if requested
    time_label = nothing
    if !isnothing(time_value)
        if isa(time_value, Number)
            time_label = Label(fig[3, :], @sprintf "t = %.2f" time_value)
        elseif isa(time_value, String)
            time_label = Label(fig[3, :], @sprintf "t = %s" time_value)
        else
            time_label = Label(fig[3, :], "")
        end
        colsize!(fig.layout, 1, Auto(false))
    end

    if singleax
        axs = strainax
    else
        axs = [strainax, resax]
    end

    (fig, strain_lines, resource_lines, time_label)
end

"""
Update the line plots with new data
"""
function update_1dsmmicrm_lines!(strain_lines, resource_lines, u, Ns)
    # Update strain lines
    for (i, line) in enumerate(strain_lines)
        xs = first.(line[1][])  # Get x coordinates from current points
        line[1][] = Point2f.(xs, u[i, :])
    end

    # Update resource lines
    for (i, line) in enumerate(resource_lines)
        xs = first.(line[1][])  # Get x coordinates from current points
        line[1][] = Point2f.(xs, u[Ns+i, :])
    end
end

"""
Plot a single snapshot of a 1D SMMiCRM solution
"""
function plot_1dsmmicrm_sol_snap(params::AbstractSMMiCRMParams, snap_u, t=nothing; singleax=false, plote=false)
    len = size(snap_u)[2]
    xs = get_space(params).dx[1] .* (0:(len-1))

    # Setup the figure
    fig, strain_lines, resource_lines, _ = setup_1dsmmicrm_figure(
        params, xs;
        singleax=singleax,
        plote=plote,
        time_value=t
    )

    # Update with data
    Ns = get_Ns(params.mmicrm_params)[1]
    update_1dsmmicrm_lines!(strain_lines, resource_lines, snap_u, Ns)

    fig
end
function plot_1dsmmicrm_sol_snap(sol, t; kwargs...)
    if isa(t, Integer)
        if t < 0
            t = length(sol.t) + t + 1
        end
        u = sol.u[t]
        t = sol.t[t]
    else
        u = sol(t)
    end
    plot_1dsmmicrm_sol_snap(sol.prob.p, u, t; kwargs...)
end
export plot_1dsmmicrm_sol_snap

"""
Create an interactive plot of a 1D SMMiCRM solution with a time slider
"""
function plot_1dsmmicrm_sol_interactive(sol; singleax=false, plote=false)
    params = sol.prob.p
    if !isa(params, AbstractSMMiCRMParams)
        throw(ArgumentError("This function can only plot solutions of SMMiCRM problems"))
    end

    len = size(sol.u[1])[2]
    xs = get_space(params).dx[1] .* (0:(len-1))
    Ns = get_Ns(params.mmicrm_params)[1]

    # Setup the figure
    fig, strain_lines, resource_lines, _ = setup_1dsmmicrm_figure(
        params, xs;
        singleax=singleax,
        plote=plote
    )

    # Add slider
    slider_layout = fig[3, 1] = GridLayout()
    timesl = Slider(slider_layout[1, 1], range=1:length(sol.t), startvalue=1)
    time_label = Label(slider_layout[1, 2], @lift(string("t = ", round(sol.t[$(timesl.value)], digits=2))))

    # Create on value change handler for slider
    on(timesl.value) do idx
        update_1dsmmicrm_lines!(strain_lines, resource_lines, sol.u[idx], Ns)
    end

    # Add keyboard controls
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press || event.action == Keyboard.repeat
            idx = timesl.value[]
            # Move 5 frames at a time when key is held
            step = event.action == Keyboard.repeat ? 10 : 1

            if event.key == Keyboard.left
                set_close_to!(timesl, max(1, idx - step))
            elseif event.key == Keyboard.right
                set_close_to!(timesl, min(length(sol.t), idx + step))
            end
        end
        return true
    end

    # Initialize with first frame
    update_1dsmmicrm_lines!(strain_lines, resource_lines, sol.u[1], Ns)

    fig
end

export plot_1dsmmicrm_sol_interactive

# 2D
"""
Base function for creating 2D heatmap plots of SMMiCRM solutions.
Returns (fig, strain_hms, resource_hms, time_label) where the heatmaps can be updated.
"""
function setup_2dsmmicrm_heatmap_figure(params::AbstractSMMiCRMParams, x_range, y_range,
    strain_clims=nothing, resource_clims=nothing,
    extra_data=false, extra_clims=nothing;
    aspect_ratio=1.5, time_value=nothing,
    strain_colormap=:viridis, resource_colormap=:plasma, extra_colormap=strain_colormap
)
    Ns, Nr = get_Ns(params.mmicrm_params)

    fig = Figure()

    # Calculate grid layout for strains and resources
    strain_rows, strain_cols = make_grid(Ns; aspect_ratio)
    resource_rows, resource_cols = make_grid(Nr; aspect_ratio)

    # Create layout for strain heatmaps and resource heatmaps
    if !extra_data
        strain_panel = fig[1, 1] = GridLayout()
        resource_panel = fig[2, 1] = GridLayout()
        colorbar_panel = fig[1:2, 2] = GridLayout()
    else
        strain_panel = fig[1, 1] = GridLayout()
        resource_panel = fig[2, 1] = GridLayout()
        extra_panel = fig[1, 2] = GridLayout()
        colorbar_panel = fig[2, 2] = GridLayout()
    end

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

        extra_kwargs = []
        if !isnothing(strain_clims)
            push!(extra_kwargs, :colorrange => strain_clims)
        end
        hm = heatmap!(ax, x_range, y_range, zeros(length(x_range), length(y_range));  # placeholder data
            colormap=strain_colormap, extra_kwargs...
        )
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

        extra_kwargs = []
        if !isnothing(resource_clims)
            push!(extra_kwargs, :colorrange => resource_clims)
        end
        hm = heatmap!(ax, x_range, y_range, zeros(length(x_range), length(y_range));  # placeholder data
            colormap=resource_colormap, extra_kwargs...
        )
        push!(resource_hms, hm)
        ax.title = "Resource $i"
        ax.xlabel = "x"
        ax.ylabel = "y"
    end

    # Maybe create an extra heatmap for the extra data
    if extra_data
        extra_ax = Axis(extra_panel[1, 1]; aspect=DataAspect())

        extra_hm = heatmap!(extra_ax, x_range, y_range, zeros(length(x_range), length(y_range));  # placeholder data
            colorrange=extra_clims, colormap=extra_colormap)
        extra_ax.title = "Extra data"
        extra_ax.xlabel = "x"
        extra_ax.ylabel = "y"
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
    if extra_data
        linkxaxes!(strain_axes[1], extra_ax)
        linkyaxes!(strain_axes[1], extra_ax)
    end

    # Add colorbars
    if !extra_data
        if !isnothing(strain_clims)
            Colorbar(colorbar_panel[1, 1], strain_hms[1], label="Strain concentrations")
        else
            for i in 1:length(strain_hms)
                Colorbar(colorbar_panel[1, i], strain_hms[i], label="Strain $i concentration")
            end
        end
        if !isnothing(resource_clims)
            Colorbar(colorbar_panel[2, 1], resource_hms[1], label="Resource concentrations")
        else
            for i in 1:length(resource_hms)
                Colorbar(colorbar_panel[2, i], resource_hms[i], label="Resource $i concentration")
            end
        end
    else
        if !isnothing(strain_clims)
            Colorbar(colorbar_panel[1, 1], strain_hms[1], label="Strain concentrations")
        else
            for i in 1:length(strain_hms)
                Colorbar(colorbar_panel[1, i], strain_hms[i], label="Strain $i concentration")
            end
        end
        num_st_hms = length(strain_hms)
        if !isnothing(resource_clims)
            Colorbar(colorbar_panel[1, num_st_hms+1], resource_hms[1], label="Resource concentrations")
        else
            for i in 1:length(resource_hms)
                Colorbar(colorbar_panel[2, num_st_hms+i], resource_hms[i], label="Resource $i concentration")
            end
        end

        Colorbar(colorbar_panel[1, 3], extra_hm, label="Extra data")
    end

    # fix the layout
    if !extra_data
        colsize!(fig.layout, 1, Auto(false))
    else
        colsize!(fig.layout, 1, Auto(false))
        colsize!(fig.layout, 2, Auto(false))
    end

    if !extra_data
        (fig, strain_hms, resource_hms, time_label)
    else
        (fig, strain_hms, resource_hms, extra_hm, time_label)
    end
end
function update_2dsmmicrm_heatmaps!(strain_hms, resource_hms, u, Ns, cell_size=1.0, extra_hm_data=nothing)
    # Update strain heatmaps
    for (i, hm) in enumerate(strain_hms)
        hm[3][] = permutedims(u[i, :, :]) # .* cell_size
    end

    # Update resource heatmaps
    for (i, hm) in enumerate(resource_hms)
        hm[3][] = permutedims(u[Ns+i, :, :]) # .* cell_size
    end

    if !isnothing(extra_hm_data)
        extra_hm_data[1][3][] = permutedims(extra_hm_data[2][:, :])
    end
end

function plot_2dsmmicrm_sol_snap_heatmap(params::AbstractSMMiCRMParams, u, t=nothing;
    aspect_ratio=1.5,
    do_strain_clims=true, do_resource_clims=true,
    kwargs...
)
    Ns = get_Ns(params.mmicrm_params)[1]

    # Create colormap ranges for better visualization
    strain_clims = do_strain_clims ? (minimum(u[1:Ns, :, :]), maximum(u[1:Ns, :, :])) : nothing
    resource_clims = do_resource_clims ? (minimum(u[Ns+1:end, :, :]), maximum(u[Ns+1:end, :, :])) : nothing

    # Setup the figure
    space_ranges = get_u_axes(u, get_space(params).dx)
    fig, strain_hms, resource_hms, _ = setup_2dsmmicrm_heatmap_figure(
        params, space_ranges[1], space_ranges[2],
        strain_clims, resource_clims;
        aspect_ratio=aspect_ratio, time_value=t, kwargs...
    )

    cell_size = space_cell_size(get_space(params))

    # Update with data
    update_2dsmmicrm_heatmaps!(strain_hms, resource_hms, u, Ns, cell_size)

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

function plot_2dsmmicrm_sol_interactive_heatmap(sol, extra_data=nothing;
    aspect_ratio=1.5,
    do_strain_clims=true, do_resource_clims=true,
    kwargs...
)
    params = sol.prob.p
    if !isa(params, AbstractSMMiCRMParams)
        throw(ArgumentError("this func can only plot solutions of SMMiCRM problems"))
    end
    Ns = get_Ns(params.mmicrm_params)[1]

    # Create colormap ranges for better visualization
    strain_clims = if do_strain_clims
        (minimum(minimum(u[1:Ns, :, :]) for u in sol.u),
            maximum(maximum(u[1:Ns, :, :]) for u in sol.u))
    else
        nothing
    end
    resource_clims = if do_resource_clims
        (minimum(minimum(u[Ns+1:end, :, :]) for u in sol.u),
            maximum(maximum(u[Ns+1:end, :, :]) for u in sol.u))
    else
        nothing
    end

    extra_clims = if !isnothing(extra_data)
        (minimum(minimum, extra_data), maximum(maximum, extra_data))
    else
        nothing
    end

    # Setup the figure
    space_ranges = get_u_axes(sol.u[1], get_space(params).dx)
    setup_rslt = setup_2dsmmicrm_heatmap_figure(
        params, space_ranges[1], space_ranges[2],
        strain_clims, resource_clims, !isnothing(extra_data), extra_clims;
        aspect_ratio, kwargs...
    )
    if isnothing(extra_data)
        fig, strain_hms, resource_hms, _ = setup_rslt
    else
        fig, strain_hms, resource_hms, extra_hm, _ = setup_rslt
    end

    # Add slider
    slider_layout = fig[3, 1] = GridLayout()
    timesl = Slider(slider_layout[1, 1], range=1:length(sol.t), startvalue=1)
    time_label = Label(slider_layout[1, 2], @lift(string("t = ", round(sol.t[$(timesl.value)], digits=2))))

    cell_size = space_cell_size(get_space(params))

    # Create on value change handler for slider
    on(timesl.value) do idx
        if isnothing(extra_data)
            update_2dsmmicrm_heatmaps!(
                strain_hms, resource_hms, sol.u[idx], Ns, cell_size
            )
        else
            update_2dsmmicrm_heatmaps!(
                strain_hms, resource_hms, sol.u[idx], Ns, cell_size,
                (extra_hm, extra_data[idx])
            )
        end
    end

    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press || event.action == Keyboard.repeat
            idx = timesl.value[]
            # Move 5 frames at a time when key is held
            step = event.action == Keyboard.repeat ? 10 : 1

            if event.key == Keyboard.left
                set_close_to!(timesl, max(1, idx - step))
            elseif event.key == Keyboard.right
                set_close_to!(timesl, min(length(sol.t), idx + step))
            end
        end
        return true
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

function plot_2dsmmicrm_sol_animation_heatmap(sol, filename=datadir(randname() * ".mp4");
    size=(600, 400), fps=30, duration=10,
    aspect_ratio=1.5,
    do_strain_clims=true, do_resource_clims=true,
    kwargs...
)
    params = sol.prob.p
    if !isa(params, AbstractSMMiCRMParams)
        throw(ArgumentError("this func can only plot solutions of SMMiCRM problems"))
    end
    Ns = get_Ns(params.mmicrm_params)[1]

    # Calculate total frames based on fps and duration
    total_frames = fps * duration
    time_indices = round.(Int, range(1, length(sol.t), length=total_frames))


    # Create colormap ranges for better visualization
    strain_clims = if do_strain_clims
        (minimum(minimum(u[1:Ns, :, :]) for u in sol.u),
            maximum(maximum(u[1:Ns, :, :]) for u in sol.u))
    else
        nothing
    end
    resource_clims = if do_resource_clims
        (minimum(minimum(u[Ns+1:end, :, :]) for u in sol.u),
            maximum(maximum(u[Ns+1:end, :, :]) for u in sol.u))
    else
        nothing
    end

    # Setup the figure
    space_ranges = get_u_axes(sol.u[1], get_space(params).dx)
    fig, strain_hms, resource_hms, time_label = setup_2dsmmicrm_heatmap_figure(
        params, space_ranges[1], space_ranges[2],
        strain_clims, resource_clims;
        aspect_ratio, time_value=@sprintf("t = %.2f", sol.t[1]),
    )

    cell_size = space_cell_size(get_space(params))

    # Initial data
    update_2dsmmicrm_heatmaps!(strain_hms, resource_hms, sol.u[1], Ns, cell_size)

    # Create animation
    framerate = fps
    record(fig, filename, time_indices; framerate=framerate) do frame_idx
        # Update time label
        time_label.text = @sprintf("t = %.2f", sol.t[frame_idx])

        # Update heatmaps
        update_2dsmmicrm_heatmaps!(strain_hms, resource_hms, sol.u[frame_idx], Ns, cell_size)
    end

    fig
end
export plot_2dsmmicrm_sol_animation_heatmap
