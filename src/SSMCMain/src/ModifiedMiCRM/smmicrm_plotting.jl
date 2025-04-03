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
        lines!(strainax, ts, getindex.(avgs, i); label=@sprintf "str %d" i)
    end
    for a in 1:Nr
        lines!(resax, ts, getindex.(avgs, Ns + a); label=@sprintf "res %d" a)
    end
    if plote
        lines!(eax, ts, energies; label=L"\epsilon")
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
function plot_2dsmmicrm_sol_snap_heatmap(params::SMMiCRMParams, u, t=nothing; aspect_ratio=1.5)
    Ns, Nr = get_Ns(params.mmicrm_params)

    fig = Figure()

    # Calculate grid layout for strains and resources
    strain_rows, strain_cols = make_grid(Ns; aspect_ratio)
    resource_rows, resource_cols = make_grid(Nr; aspect_ratio)

    # Create layout for strain heatmaps and resource heatmaps
    strain_panel = fig[1, 1] = GridLayout()
    resource_panel = fig[2, 1] = GridLayout()
    colorbar_panel = fig[1:2, 2] = GridLayout()

    # Add time label
    if !isnothing(t)
        if isa(t, Number)
            Label(fig[3, 1], (@sprintf "t = %.2f" t))
        elseif isa(t, String)
            Label(fig[3, 1], (@sprintf "t = %s" t))
        end
    end

    # Create axes for strains and resources using the calculated grid
    strain_axes = Matrix{Union{Axis,Nothing}}(nothing, strain_rows, strain_cols)
    resource_axes = Matrix{Union{Axis,Nothing}}(nothing, resource_rows, resource_cols)

    # Create colormap ranges for better visualization
    strain_clims = (minimum(u[1:Ns, :, :]), maximum(u[1:Ns, :, :]))
    resource_clims = (minimum(u[Ns+1:end, :, :]), maximum(u[Ns+1:end, :, :]))

    # Create heatmaps for each strain
    strain_hms = []
    for i in 1:Ns
        row = div(i - 1, strain_cols) + 1
        col = mod(i - 1, strain_cols) + 1
        ax = Axis(strain_panel[row, col])
        strain_axes[row, col] = ax

        hm = heatmap!(ax, permutedims(u[i, :, :]);
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
        ax = Axis(resource_panel[row, col])
        resource_axes[row, col] = ax

        hm = heatmap!(ax, permutedims(u[Ns+i, :, :]);
            colorrange=resource_clims, colormap=:plasma)
        push!(resource_hms, hm)
        ax.title = "Resource $i"
        ax.xlabel = "x"
        ax.ylabel = "y"
    end

    # Add colorbars
    Colorbar(colorbar_panel[1, 1], strain_hms[1], label="Strain concentration")
    Colorbar(colorbar_panel[2, 1], resource_hms[1], label="Resource concentration")

    # fix the layout
    colsize!(fig.layout, 1, Auto(false))

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
    Ns, Nr = get_Ns(params.mmicrm_params)

    fig = Figure()

    # Calculate grid layout for strains and resources
    strain_rows, strain_cols = make_grid(Ns; aspect_ratio)
    resource_rows, resource_cols = make_grid(Nr; aspect_ratio)

    # Create layout for strain heatmaps, resource heatmaps, and slider
    strain_panel = fig[1, 1] = GridLayout()
    resource_panel = fig[2, 1] = GridLayout()
    slider_layout = fig[3, 1] = GridLayout()
    colorbar_panel = fig[1:2, 2] = GridLayout()

    # Create time slider
    timesl = Slider(slider_layout[1, 1], range=1:length(sol.t), startvalue=1)
    time_label = Label(slider_layout[1, 2], @lift(string("t = ", round(sol.t[$(timesl.value)], digits=2))))

    # Create axes for strains and resources using the calculated grid
    strain_axes = Matrix{Union{Axis,Nothing}}(nothing, strain_rows, strain_cols)
    resource_axes = Matrix{Union{Axis,Nothing}}(nothing, resource_rows, resource_cols)

    # Create colormap ranges for better visualization
    strain_clims = (minimum(minimum(u[1:Ns, :, :]) for u in sol.u),
        maximum(maximum(u[1:Ns, :, :]) for u in sol.u))
    resource_clims = (minimum(minimum(u[Ns+1:end, :, :]) for u in sol.u),
        maximum(maximum(u[Ns+1:end, :, :]) for u in sol.u))

    # Create and update heatmaps for each strain
    strain_hms = []
    for i in 1:Ns
        row = div(i - 1, strain_cols) + 1
        col = mod(i - 1, strain_cols) + 1
        ax = Axis(strain_panel[row, col])
        strain_axes[row, col] = ax

        heatmap_data = @lift begin
            timestep_data = sol.u[$(timesl.value)][i, :, :]
            permutedims(timestep_data)
        end

        hm = heatmap!(ax, heatmap_data; colorrange=strain_clims, colormap=:viridis)
        push!(strain_hms, hm)
        ax.title = "Strain $i"
        ax.xlabel = "x"
        ax.ylabel = "y"
    end

    # Create and update heatmaps for each resource
    resource_hms = []
    for i in 1:Nr
        row = div(i - 1, resource_cols) + 1
        col = mod(i - 1, resource_cols) + 1
        ax = Axis(resource_panel[row, col])
        resource_axes[row, col] = ax

        heatmap_data = @lift begin
            timestep_data = sol.u[$(timesl.value)][Ns+i, :, :]
            permutedims(timestep_data)
        end

        hm = heatmap!(ax, heatmap_data; colorrange=resource_clims, colormap=:plasma)
        push!(resource_hms, hm)
        ax.title = "Resource $i"
        ax.xlabel = "x"
        ax.ylabel = "y"
    end
    #
    # link all axes to the first
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

    fig
end
export plot_2dsmmicrm_sol_interactive_heatmap

function plot_2dsmmicrm_sol_animation(sol, filename=datadir(randname() * ".mp4"); size=(600, 400), fps=30, duration=10, aspect_ratio=1.5)
    params = sol.prob.p
    if !isa(params, SMMiCRMParams)
        throw(ArgumentError("this func can only plot solutions of SMMiCRM problems"))
    end
    Ns, Nr = get_Ns(params.mmicrm_params)

    # Calculate total frames based on fps and duration
    total_frames = fps * duration
    time_indices = round.(Int, range(1, length(sol.t), length=total_frames))

    fig = Figure(; size=size)

    # Calculate grid layout for strains and resources
    strain_rows, strain_cols = make_grid(Ns; aspect_ratio)
    resostrain_rows, strain_cols = make_grid(Ns; aspect_ratio)

    # Create layout for strain and resource heatmaps
    strain_panel = fig[1, 1] = GridLayout()
    resource_panel = fig[2, 1] = GridLayout()
    colorbar_panel = fig[1:2, 2] = GridLayout()
    time_label = Label(fig[3, 1], @sprintf("t = %.2f", sol.t[1]))

    # Create axes for strains and resources using the calculated grid
    strain_axes = Matrix{Union{Axis,Nothing}}(nothing, strain_rows, strain_cols)
    resource_axes = Matrix{Union{Axis,Nothing}}(nothing, resource_rows, resource_cols)

    # Create colormap ranges for better visualization
    strain_clims = (minimum(minimum(u[1:Ns, :, :]) for u in sol.u),
        maximum(maximum(u[1:Ns, :, :]) for u in sol.u))
    resource_clims = (minimum(minimum(u[Ns+1:end, :, :]) for u in sol.u),
        maximum(maximum(u[Ns+1:end, :, :]) for u in sol.u))

    # Create heatmaps for each strain
    strain_hms = []
    for i in 1:Ns
        row = div(i - 1, strain_cols) + 1
        col = mod(i - 1, strain_cols) + 1
        ax = Axis(strain_panel[row, col])
        strain_axes[row, col] = ax

        hm = heatmap!(ax, permutedims(sol.u[1][i, :, :]);
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
        ax = Axis(resource_panel[row, col])
        resource_axes[row, col] = ax

        hm = heatmap!(ax, permutedims(sol.u[1][Ns+i, :, :]);
            colorrange=resource_clims, colormap=:plasma)
        push!(resource_hms, hm)
        ax.title = "Resource $i"
        ax.xlabel = "x"
        ax.ylabel = "y"
    end

    # Add colorbars
    Colorbar(colorbar_panel[1, 1], strain_hms[1], label="Strain concentration")
    Colorbar(colorbar_panel[2, 1], resource_hms[1], label="Resource concentration")

    # fix the layout
    colsize!(fig.layout, 1, Auto(false))

    # Create animation
    framerate = fps
    record(fig, filename, time_indices; framerate=framerate) do frame_idx
        # Update time label
        time_label.text = @sprintf("t = %.2f", sol.t[frame_idx])

        # Update strain heatmaps
        for (i, hm) in enumerate(strain_hms)
            hm[3][] = permutedims(sol.u[frame_idx][i, :, :])
        end

        # Update resource heatmaps
        for (i, hm) in enumerate(resource_hms)
            hm[3][] = permutedims(sol.u[frame_idx][Ns+i, :, :])
        end
    end

    fig
end
export plot_2dsmmicrm_sol_animation

################################################################################
# Specific plots which are not general but still useful!
################################################################################
