using SSMCMain, SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.MinimalModelSemisymbolic

using Base.Threads, OhMyThreads
using JLD2

using CairoMakie


function add_peak_2d!(data, dx, center, width, mag=1.0)
    sx, sy = size(data)
    centers = [center,
        (center[1] - sx, center[2]),
        (center[1] + sx, center[2]),
        (center[1], center[2] - sy),
        (center[1], center[2] + sy)
    ]

    sigma = width / 2.355
    a = mag / (sigma * sqrt(2 * pi))
    b = 2 * sigma^2

    for i in 1:sx
        for j in 1:sy
            for (cx, cy) in centers
                dist = sqrt((abs(i - cx) * dx[1])^2 + (abs(j - cy) * dx[2])^2)
                peak_val = a * exp(-dist^2 / b)
                data[i, j] += peak_val
            end
        end
    end
    data
end

function add_peaks_2d!(data, dx, num_peaks, width, mag=1.0)
    sx, sy = size(data)
    for _ in 1:num_peaks
        cx = rand(1:sx)
        cy = rand(1:sy)
        add_peak_2d!(data, dx, (cx, cy), width, mag)
    end
    data
end

function make_cosmo_mmicrm_params(m, l, K, c, d)
    D = fill(0.0, 2, 3, 3)
    D[1, :, :] .= [0.0 0.0 0.0; 0.0 0.0 0.0; 1 0.0 0.0]
    D[2, :, :] .= [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
    make_sammicrmparams(2, 3;
        D=SArray{Tuple{2,3,3}}(D),
        K=[K, 0.0, 0.0],
        r=[1.0, 1.0, 1.0],
        l=[l 0.0 0.0; l 0.0 0.0],
        c=[c d 0.0; c 0.0 d],
        m=[m, m],
    )
end

function main()
    # Choose which params to use
    mmicrm_params = make_cosmo_mmicrm_params(0.1, 1.0, 2.0, 0.5, 1.0)
    Ds = SA[1e-5, 1e-5, 5.0, 1.0, 1.0]

    # Number of random simulations to run
    num_runs = 2
    out_basename = "./out/run_"

    # Specify the system size and time
    sys_size = (800, 800)
    dx = SA[0.01, 0.01]
    T = 1e10

    # Choose the random initialization
    num_peaks = 20
    peak_width = 0.3 # sys_size[1] * dx[1] / 20
    peak_mag = 1.0

    # initiate with glucose at max, and no N or R, add peaks on top of this later
    base_u0 = expand_u0_to_size(sys_size, [0.0, 0.0, mmicrm_params.K[1], 0.0, 0.0])

    # prep the params with periodic BCs, threaded
    spatial_params = SASMMiCRMParams(
        mmicrm_params,
        Ds,
        make_cartesianspace_smart(2; dx, bcs=Periodic()),
        nthreads()
    )

    for i in 1:num_runs
        @printf "Starting run %d\n" i
        flush(stdout)

        # but add a bunch of peaks of N to the base_u0
        u0 = copy(base_u0)
        add_peaks_2d!((@view u0[1, :, :]), dx, num_peaks, peak_width, peak_mag)
        add_peaks_2d!((@view u0[2, :, :]), dx, num_peaks, peak_width, peak_mag)

        # make and solve the problem
        sp = make_smmicrm_problem(spatial_params, u0, T)
        @time sps = solve(sp, QNDF();
            callback=make_timer_callback(60 * 60 * 24),
            dense=false
        )
        print_spatial_solution_stats(sps)

        # save the solution
        save_object(out_basename * string(i) * ".jld2", sps)

        # save an image of the final state
        final_state_fig = plot_2dsmmicrm_sol_snap_heatmap(sps, -1)
        Makie.save(out_basename * string(i) * ".png", final_state_fig)

        # and add an animation for good measure
        plot_2dsmmicrm_sol_animation_heatmap(sps, out_basename * string(i) * ".mp4")

        @printf "Finished run %d\n" i
        flush(stdout)
    end
end

function ltest()
    # Choose which params to use
    mmicrm_params = make_cosmo_mmicrm_params(0.1, 1.0, 2.0, 0.5, 1.0)
    Ds = SA[1e-5, 1e-5, 5.0, 1.0, 1.0]

    # Number of random simulations to run
    num_runs = 2
    out_basename = "./out/run_"

    # Specify the system size and time
    sys_size = (100, 100)
    dx = SA[0.01, 0.01]
    T = 1e10

    # Choose the random initialization
    num_peaks = 20
    peak_width = 0.3 # sys_size[1] * dx[1] / 20
    peak_mag = 1.0

    # initiate with glucose at max, and no N or R, add peaks on top of this later
    base_u0 = expand_u0_to_size(sys_size, [0.0, 0.0, mmicrm_params.K[1], 0.0, 0.0])

    # prep the params with periodic BCs, threaded
    spatial_params = SASMMiCRMParams(
        mmicrm_params,
        Ds,
        make_cartesianspace_smart(2; dx, bcs=Periodic()),
        nthreads()
    )

    for i in 1:num_runs
        @printf "Starting run %d\n" i
        flush(stdout)

        # but add a bunch of peaks of N to the base_u0
        u0 = copy(base_u0)
        add_peaks_2d!((@view u0[1, :, :]), dx, num_peaks, peak_width, peak_mag)
        add_peaks_2d!((@view u0[2, :, :]), dx, num_peaks, peak_width, peak_mag)

        # make and solve the problem
        sp = make_smmicrm_problem(spatial_params, u0, T)
        @time sps = solve(sp, QNDF();
            callback=make_timer_callback(60 * 60 * 24),
            dense=false
        )
        print_spatial_solution_stats(sps)

        # save the solution
        save_object(out_basename * string(i) * ".jld2", sps)

        # save an image of the final state
        final_state_fig = plot_2dsmmicrm_sol_snap_heatmap(sps, -1)
        Makie.save(out_basename * string(i) * ".png", final_state_fig)

        # and add an animation for good measure
        plot_2dsmmicrm_sol_animation_heatmap(sps, out_basename * string(i) * ".mp4")

        @printf "Finished run %d\n" i
        flush(stdout)
    end
end

################################################################################
# Extras
################################################################################
function add_animation(basename)
    sps = load_object(basename * ".jld2")
    print_spatial_solution_stats(sps)
    plot_2dsmmicrm_sol_animation_heatmap(sps, basename * ".mp4")
end

function add_animations_job()
    out_basename = "./out/run_"
    for i in 4:11
        @printf "Starting on %d\n" i
        flush(stdout)

        basename = out_basename * string(i)
        add_animation(basename)

        @printf "Finished %d\n" i
        flush(stdout)
    end
end
