function plot_many_final_states(f, num_per_K, Kis=nothing)
    fmd = f["metadata"]
    results = f["results"]
    tmm_params = f["tmm_params"]
    Ks = fmd.Ks
    dx = fmd.L / fmd.sN
    Ds = SA[fmd.DN1, fmd.DN2, fmd.DI, fmd.DR1, fmd.DR2];

    if isnothing(Kis)
        Kis = 1:length(Ks)
    end

    fig = Figure(;
        size=(400 * num_per_K, 300 * length(Kis))
    )

    for Kii in 1:length(Kis)
        Ki = Kis[Kii]
        Label(fig[Kii,0], (@sprintf "K=%.3g" Ks[Ki]);
            fontsize=20,
            tellheight=false,
            rotation=pi/2,
        )
        tmm_ps = tmm_params[Ki]
        sps = BSMMiCRMParams(tmmsp_to_mmicrm(tmm_ps), Ds, CartesianSpace{1,Tuple{Periodic}}(SA[dx]))
        for ri in 1:num_per_K
            fs = results[Ki].fss[ri]
            rc = results[Ki].retcodes[ri]
            axs, axr = plot_spatial_fs!(fig[Kii, ri], fs, 2, fmd.sN, dx)
            mresid = smmicrmmaxresid(fs, sps)
            # axs.title = "run $ri, retcode $rc, maxresid "
            axs.title = @sprintf "run %d, retcode %s, maxresid %.3g" ri string(rc) mresid
        end
    end

    fig
end
