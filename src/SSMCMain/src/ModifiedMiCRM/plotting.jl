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

    FigureAxisAnything(fig, [strainax, resax], nothing)
end
export plot_mmicrm_sol

function plot_linstab_lambdas(ks, lambdas;
    figure=(;),
    axis=(;),
    legend=true,
    imthreshold=1e-8,
)
    fig = Figure(; figure...)
    ax = Axis(fig[1, 1]; axis...)

    num_lambdas = length(lambdas[1])

    for li in 1:num_lambdas
        # ls = [lambdas[i][li] for i in 1:length(lambdas)]
        ls = getindex.(lambdas, li)

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
    if legend
        axislegend(ax)
    end

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
function plot_1dsmmicrm_sol_snap(params::AbstractSMMiCRMParams, snap_u, t=nothing;
    singleax=false, plote=false,
    dx=get_space(params).dx[1],
)
    len = size(snap_u)[2]
    xs = dx .* ((1:len) .- 0.5)

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
    xs = get_space(params).dx[1] .* ((1:len) .- 0.5)
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

# Simpler version of plotting a 1D spatial solution
function get_spatial_gridpoints_dx(sN::Integer, dx)
    ((1:sN) .- 0.5) .* dx
end
function get_spatial_gridpoints_dx(u::AbstractVector, dx)
    get_spatial_gridpoints_L(length(u), dx)
end
function get_spatial_gridpoints_L(N::Integer, L)
    get_spatial_gridpoints_dx(N, L / N)
end
function get_spatial_gridpoints_L(u::AbstractVector, L)
    get_spatial_gridpoints_L(length(u), L)
end
export get_spatial_gridpoints_dx, get_spatial_gridpoints_L

function plot_spatial_fs!(where, u, Ns, sN, dx, ss=nothing;
    axis=(;),
    scolor=i -> Cycled(i),
    rcolor=i -> Cycled(Ns + i),
)
    Nr = size(u)[1] - Ns
    xs = get_spatial_gridpoints_dx(sN, dx)

    gl = GridLayout(where)

    axs = Axis(gl[1, 1]; axis...)
    axr = Axis(gl[2, 1]; axis...)
    linkxaxes!(axs, axr)
    hidexdecorations!(axs)
    rowgap!(gl, 4.0)

    for i in 1:Ns
        lines!(axs, xs, u[i, :]; color=scolor(i))
    end
    for a in 1:Nr
        lines!(axr, xs, u[Ns+a, :]; color=rcolor(a))
    end

    if !isnothing(ss)
        for i in 1:Ns
            hlines!(axs, ss[i];
                color=scolor(i),
                linestyle=:dash
            )
        end
        for a in 1:Nr
            hlines!(axr, ss[Ns+a];
                color=rcolor(a),
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
    plot_spatial_fs!(fig[1, 1], args...; kwargs...)

    fig
end
export plot_spatial_fs!, plot_spatial_fs

################################################################################
# Space-time plots of 1D many strain solutions
################################################################################
"""
    centres_to_edges(cs; closed=true, geometric=false)

`length(cs)+1` cell edges from `length(cs)` cell centres, the way `heatmap` wants
them. Interior edges sit at the midpoint of neighbouring centres, the geometric
mean of them if `geometric` (the midpoint an axis drawn on a log scale shows).
When `closed` the two outer edges are the outermost centres themselves so the plot
never extends past the data - this is what keeps the first edge of a log time axis
positive, at the cost of drawing the first and last cells at half width. Otherwise
they are extrapolated half a cell outwards, which for `geometric` also stays
positive but can drag a log axis a long way down if the first two points are close
together.
"""
function centres_to_edges(cs; closed=true, geometric=false)
    n = length(cs)
    if n < 2
        throw(ArgumentError("need at least 2 centres to define edges"))
    end
    mid(a, b) = geometric ? sqrt(a * b) : (a + b) / 2

    es = zeros(float(eltype(cs)), n + 1)
    for i in 1:(n-1)
        es[i+1] = mid(cs[i], cs[i+1])
    end
    if closed
        es[1] = cs[1]
        es[end] = cs[end]
    else
        es[1] = geometric ? cs[1]^2 / es[2] : 2 * cs[1] - es[2]
        es[end] = geometric ? cs[end]^2 / es[end-1] : 2 * cs[end] - es[end-1]
    end
    es
end
export centres_to_edges

"""
    log_spaced_indices(xs, n)

`n` indices into the increasing, strictly positive `xs`, picked so the values kept
are as evenly spaced in log space as the data allows. The first and last are
always kept, and `n === nothing` (or an `n` past `length(xs)`) keeps everything.

Whatever budget is left over after that - `xs` being far from log spaced itself
means several evenly spaced targets can land on the same point - is then spent by
repeatedly splitting whichever log gap between neighbouring kept values is
currently the widest. So the full `n` points always get used, with the extra ones
going where `xs` is densest in log space rather than being dropped.
"""
function log_spaced_indices(xs, n)
    N = length(xs)
    if isnothing(n) || (N <= n)
        return collect(1:N)
    end
    if n < 2
        throw(ArgumentError("cannot keep fewer than 2 points"))
    end

    ls = log.(xs)

    # the closest thing to evenly spaced in log space that the data allows
    is = Int[]
    for target in range(ls[1], ls[end], n)
        j = clamp(searchsortedfirst(ls, target), 1, N)
        if (j > 1) && ((ls[j] - target) > (target - ls[j-1]))
            j -= 1
        end
        push!(is, j)
    end
    unique!(is)

    while length(is) < n
        besti, bestgap, bestj = 0, 0.0, 0
        for k in 1:(length(is)-1)
            a, b = is[k], is[k+1]
            b > a + 1 || continue # nothing left in between to add
            gap = ls[b] - ls[a]
            gap > bestgap || continue

            # whichever of the points in between sits closest to the middle
            mid = (ls[a] + ls[b]) / 2
            j = clamp(searchsortedfirst(ls, mid), a + 1, b - 1)
            if (j > a + 1) && ((ls[j] - mid) > (mid - ls[j-1]))
                j -= 1
            end

            besti, bestgap, bestj = k, gap, j
        end
        besti == 0 && break
        insert!(is, besti + 1, bestj)
    end

    is
end
export log_spaced_indices

# nb contiguous, as close to equal as possible, ranges covering 1:n
function even_blocks(n, nb)
    [(round(Int, (b - 1) * n / nb)+1):round(Int, b * n / nb) for b in 1:nb]
end

"""
    spacetime_data(ts, us, Ns, dx; kwargs...)
    spacetime_data(sol::ODESolution; kwargs...)

Boil a 1D solution (`us` being the states at `ts`, each fields x space) down to
what a space-time heatmap of its strains needs, as
`(; tedges, xedges, ts, strains)`. `strains` is an `Ns x nx x nt` array of strain
concentrations and the two edge vectors are ready to be passed straight to
`heatmap`.

Times not strictly greater than `tmin` (`0` by default) are dropped as they cannot
go on a log axis. The grid is then downsampled to at most `max_nt` times, picked
evenly in log time to match how they get drawn, and `max_nx` spatial cells, each
the average of a contiguous block of the original ones. Pass `nothing` for either
to keep the full resolution - the defaults are only meant to keep the number of
heatmap cells in the range a laptop can render.
"""
function spacetime_data(ts, us, Ns, dx;
    tmin=nothing, max_nt=1500, max_nx=1000, quiet=false
)
    if length(ts) != length(us)
        throw(ArgumentError(@sprintf "got %d times but %d states" length(ts) length(us)))
    end
    if !issorted(ts)
        throw(ArgumentError("times must be increasing"))
    end

    # drop what cannot go on a log axis
    lo = isnothing(tmin) ? zero(eltype(ts)) : tmin
    keep = findall(>(lo), ts)
    if length(keep) < 2
        throw(ArgumentError(@sprintf "only %d of the times are greater than %g, need at least 2" length(keep) lo))
    end
    ts = ts[keep]
    us = us[keep]

    sN = size(us[1], 2)
    tis = log_spaced_indices(ts, max_nt)
    blocks = even_blocks(sN, isnothing(max_nx) ? sN : min(sN, max_nx))

    if !quiet && ((length(tis) < length(ts)) || (length(blocks) < sN))
        @info @sprintf "downsampled the space-time grid from %d x %d to %d x %d (space x time)" sN length(ts) length(blocks) length(tis)
    end

    strains = Array{float(eltype(us[1])),3}(undef, Ns, length(blocks), length(tis))
    for (ti, si) in enumerate(tis)
        u = us[si]
        for (xi, block) in enumerate(blocks)
            for i in 1:Ns
                strains[i, xi, ti] = mean(view(u, i, block))
            end
        end
    end

    (;
        tedges=centres_to_edges(ts[tis]; closed=true, geometric=true),
        xedges=[zero(dx); [last(block) * dx for block in blocks]],
        ts=ts[tis],
        strains
    )
end
function spacetime_data(sol::ODESolution; kwargs...)
    spacetime_data(spacetime_solargs(sol)...; kwargs...)
end
export spacetime_data

# (ts, us, Ns, dx) out of a 1D SMMiCRM solution
function spacetime_solargs(sol::ODESolution)
    params = sol.prob.p
    if !isa(params, AbstractSMMiCRMParams)
        throw(ArgumentError("this func can only plot solutions of SMMiCRM problems"))
    end
    if ndims(params) != 1
        throw(ArgumentError("this func can only plot 1D solutions of SMMiCRM problems"))
    end
    Ns, _ = get_Ns(params)
    (sol.t, sol.u, Ns, get_space(params).dx[1])
end

const SPACETIME_AXIS_DEFAULTS = (;
    yscale=log10,
    xlabel="Space",
    ylabel="Time",
    xgridvisible=false,
    ygridvisible=false,
)

"""
    plot_spacetime_biomass!(where, data; kwargs...)
    plot_spacetime_biomass!(where, ts, us, Ns, dx; tmin=nothing, max_nt=1500, max_nx=1000, kwargs...)
    plot_spacetime_biomass!(where, sol::ODESolution; kwargs...)
    plot_spacetime_biomass(args...; figure=(;), kwargs...)

Space-time heatmap of the total biomass (all `Ns` strains summed) of a 1D
solution, time running up a log axis. Returns `(; ax, hm, cb, data)`, or the
`Figure` for the non `!` version.

`data` is what [`spacetime_data`](@ref) returns - pass it directly when the states
are too big to keep around or are being plotted more than once, otherwise hand
over the solution itself and the `tmin`, `max_nt` and `max_nx` kwargs go there.
`axis` is merged over `SPACETIME_AXIS_DEFAULTS`. With a non identity `colorscale`
(eg `log10`) the data is clamped into `colorrange`, which by default then only
covers the strictly positive values.
"""
function plot_spacetime_biomass!(where, data::NamedTuple;
    axis=(;),
    colormap=:viridis, colorscale=identity, colorrange=nothing,
    colorbar=false, colorbar_kwargs=(;)
)
    bs = dropdims(sum(data.strains; dims=1); dims=1)

    if isnothing(colorrange)
        pos = filter(>(0), vec(bs))
        hi = isempty(pos) ? 1.0 : maximum(pos)
        lo = if colorscale === identity
            0.0
        else
            isempty(pos) ? hi / 1e6 : minimum(pos)
        end
        colorrange = (lo, hi)
    end
    if !(colorrange[1] < colorrange[2]) # degenerate data, still needs a valid range
        colorrange = (colorrange[1], colorrange[1] + max(abs(colorrange[1]), 1.0))
    end
    if colorscale !== identity
        bs = clamp.(bs, colorrange...)
    end

    gl = GridLayout(where)
    ax = Axis(gl[1, 1]; merge(SPACETIME_AXIS_DEFAULTS, axis)...)
    hm = heatmap!(ax, data.xedges, data.tedges, bs; colormap, colorscale, colorrange)
    cb = if colorbar
        Colorbar(gl[1, 2], hm; merge((; label="Total biomass"), colorbar_kwargs)...)
    end

    (; ax, hm, cb, data)
end
function plot_spacetime_biomass!(where, ts, us, Ns, dx;
    tmin=nothing, max_nt=1500, max_nx=1000, quiet=false, kwargs...
)
    plot_spacetime_biomass!(where,
        spacetime_data(ts, us, Ns, dx; tmin, max_nt, max_nx, quiet); kwargs...)
end
function plot_spacetime_biomass!(where, sol::ODESolution; kwargs...)
    plot_spacetime_biomass!(where, spacetime_solargs(sol)...; kwargs...)
end
function plot_spacetime_biomass(args...; figure=(;), kwargs...)
    fig = Figure(; figure...)
    plot_spacetime_biomass!(fig[1, 1], args...; kwargs...)

    fig
end
export plot_spacetime_biomass!, plot_spacetime_biomass

_strain_color(strain_colors, i) = strain_colors[mod1(i, length(strain_colors))]
_strain_color(strain_colors::Function, i) = strain_colors(i)

function _blend_space(blend)
    if blend === :srgb
        RGB{Float64}
    elseif blend === :oklab
        Oklab{Float64}
    elseif blend === :lab
        Lab{Float64}
    else
        throw(ArgumentError(@sprintf "unknown blend space %s, should be one of :srgb, :oklab or :lab" string(blend)))
    end
end

_to_rgb(c) = mapc(x -> clamp(x, 0.0, 1.0), convert(RGB{Float64}, c))

function _lerp_color(a::C, b::C, t) where {C}
    C(comp1(a) + t * (comp1(b) - comp1(a)),
        comp2(a) + t * (comp2(b) - comp2(a)),
        comp3(a) + t * (comp3(b) - comp3(a)))
end

# componentwise weighted mean, in whatever space the colours are already given in
function _mix_in_space(xs, cols::AbstractVector{C}, q=1.0) where {C}
    tot = 0.0
    @inbounds for x in xs
        x > 0 && (tot += q == 1.0 ? x : x^q)
    end
    if !(tot > 0)
        return C(0.0, 0.0, 0.0)
    end

    c1 = c2 = c3 = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        x > 0 || continue
        w = (q == 1.0 ? x : x^q) / tot
        c = cols[i]
        c1 += w * comp1(c)
        c2 += w * comp2(c)
        c3 += w * comp3(c)
    end

    C(c1, c2, c3)
end

"""
    mix_composition_color(xs, strain_colors=ColorSchemes.tab20; blend=:srgb, q=1.0)

The colour of a community made up of `xs`: the per strain colours averaged
componentwise, weighted by relative abundance. `strain_colors` is either indexed
by strain (cycling if there are more strains than colours) or called with the
strain index.

`blend` is the space that average is taken in - `:srgb` mixes the gamma encoded
channels directly (which is what folding the strains in one at a time through a
two colour gradient does, only this does it in one pass), `:oklab` or `:lab` mix
perceptually, so equal shifts in abundance move the colour by equal perceived
amounts and mixtures do not darken or drift the way sRGB ones do.

`q` reweights the strains as `xᵢ^q` before normalising, the Hill exponent from
diversity indices. It is scale invariant for any `q`, unlike weighting by
`log(xᵢ)` - which also goes negative below 1 and diverges at 0. `q = 1` is plain
relative abundance, `q < 1` (say `0.3` to `0.5`) pulls the mix away from the
dominant strain so subdominant ones actually show, and `q > 1` sharpens towards
whoever is winning.
"""
function mix_composition_color(xs, strain_colors=ColorSchemes.tab20;
    blend=:srgb, q=1.0
)
    C = _blend_space(blend)
    cols = [convert(C, RGB{Float64}(_strain_color(strain_colors, i))) for i in eachindex(xs)]

    _to_rgb(_mix_in_space(xs, cols, q))
end
export mix_composition_color

"""
    composition_colors(strains; kwargs...)

Turn an `Ns x nx x nt` array of strain concentrations, as
[`spacetime_data`](@ref) returns, into the `nx x nt` matrix of colours where hue
shows the community composition ([`mix_composition_color`](@ref) of the local
strain mix, see there for `blend`, `q` and `strain_colors`) and the total biomass
sets how strongly that colour is shown.

`alphanorm` sets what the biomass is measured against - `:global` the largest
total anywhere in the run, `:time` the largest total across space at that same
time (which keeps the shape of the pattern visible even while everything is still
small), or `:none` to take the totals as already being on a 0 to 1 scale.
`alpha_transform` is applied to the normalised value before it is clamped into
`[0, 1]`, eg `a -> a^0.4` or `a -> log1p(99a) / log1p(99)` to lift the faint end
of a biomass that spans decades.

`biomass` then says which channel that value drives:

  - `:alpha` leaves the mixed colour alone and puts it in the opacity, so what
    ends up on screen depends on whatever the axis is drawn on top of.
  - `:lightness` interpolates from `background` to the mixed colour inside
    `blend`'s space and returns opaque colours. With `blend=:oklab` that is a
    proper bivariate map - perceptual lightness carries the biomass, hue and
    chroma the composition, with the two staying separable - and it survives
    being saved to a vector format without any transparency.
"""
function composition_colors(strains;
    blend=:srgb, biomass=:alpha, background=RGB{Float64}(1.0, 1.0, 1.0), q=1.0,
    alphanorm=:global, alpha_transform=identity, strain_colors=ColorSchemes.tab20
)
    if !(biomass in (:alpha, :lightness))
        throw(ArgumentError(@sprintf "unknown biomass channel %s, should be :alpha or :lightness" string(biomass)))
    end

    Ns, _, nt = size(strains)
    tots = dropdims(sum(strains; dims=1); dims=1)

    refs = if alphanorm === :global
        fill(maximum(tots), nt)
    elseif alphanorm === :time
        [maximum(view(tots, :, ti)) for ti in 1:nt]
    elseif alphanorm === :none
        ones(eltype(tots), nt)
    else
        throw(ArgumentError(@sprintf "unknown alphanorm %s, should be one of :global, :time or :none" string(alphanorm)))
    end

    C = _blend_space(blend)
    cols = [convert(C, RGB{Float64}(_strain_color(strain_colors, i))) for i in 1:Ns]

    _composition_colors(strains, tots, refs, cols, convert(C, RGB{Float64}(background)),
        q, biomass === :alpha, alpha_transform)
end
export composition_colors

# the pixel loop, behind a barrier so it specialises on the blend space
function _composition_colors(strains, tots, refs, cols::AbstractVector{C}, bg::C,
    q, usealpha, alpha_transform
) where {C}
    nx, nt = size(tots)
    empty = usealpha ? RGBA{Float64}(0.0, 0.0, 0.0, 0.0) : alphacolor(_to_rgb(bg), 1.0)

    out = Matrix{RGBA{Float64}}(undef, nx, nt)
    for ti in 1:nt
        ref = refs[ti]
        for xi in 1:nx
            b = tots[xi, ti]
            t = ((b > 0) && (ref > 0)) ? clamp(alpha_transform(b / ref), 0.0, 1.0) : 0.0
            if t == 0.0
                out[xi, ti] = empty
                continue
            end

            m = _mix_in_space(view(strains, :, xi, ti), cols, q)
            out[xi, ti] = if usealpha
                alphacolor(_to_rgb(m), t)
            else
                alphacolor(_to_rgb(_lerp_color(bg, m, t)), 1.0)
            end
        end
    end

    out
end

"""
    plot_spacetime_composition!(where, data; kwargs...)
    plot_spacetime_composition!(where, ts, us, Ns, dx; tmin=nothing, max_nt=1500, max_nx=1000, kwargs...)
    plot_spacetime_composition!(where, sol::ODESolution; kwargs...)
    plot_spacetime_composition(args...; figure=(;), kwargs...)

Space-time heatmap of a 1D solution showing both who is there and how much of
them, time running up a log axis: colour is the local community composition and
opacity the total biomass. Returns `(; ax, hm, data, colors)`, or the `Figure` for
the non `!` version.

Every kwarg other than `axis` is passed on to [`composition_colors`](@ref) -
`blend`, `q`, `biomass`, `alphanorm`, `alpha_transform`, `strain_colors` and
`background` - and `axis` is merged over `SPACETIME_AXIS_DEFAULTS`. `data` is what
[`spacetime_data`](@ref) returns - pass it directly when the states are too big to
keep around or are being plotted more than once, otherwise hand over the solution
itself and the `tmin`, `max_nt` and `max_nx` kwargs go there.

The default `biomass=:alpha` draws with transparency, against whatever is behind
the axis; `biomass=:lightness` (best with `blend=:oklab`) bakes the background in
and returns opaque colours instead.
"""
function plot_spacetime_composition!(where, data::NamedTuple; axis=(;), kwargs...)
    colors = composition_colors(data.strains; kwargs...)

    gl = GridLayout(where)
    ax = Axis(gl[1, 1]; merge(SPACETIME_AXIS_DEFAULTS, axis)...)
    hm = heatmap!(ax, data.xedges, data.tedges, colors)

    (; ax, hm, data, colors)
end
function plot_spacetime_composition!(where, ts, us, Ns, dx;
    tmin=nothing, max_nt=1500, max_nx=1000, quiet=false, kwargs...
)
    plot_spacetime_composition!(where,
        spacetime_data(ts, us, Ns, dx; tmin, max_nt, max_nx, quiet); kwargs...)
end
function plot_spacetime_composition!(where, sol::ODESolution; kwargs...)
    plot_spacetime_composition!(where, spacetime_solargs(sol)...; kwargs...)
end
function plot_spacetime_composition(args...; figure=(;), kwargs...)
    fig = Figure(; figure...)
    plot_spacetime_composition!(fig[1, 1], args...; kwargs...)

    fig
end
export plot_spacetime_composition!, plot_spacetime_composition
