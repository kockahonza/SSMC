import SSMCMain.ModifiedMiCRM.MinimalModelV2

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

function peak_spacings_rep1!(place, f, iK)
    fmetadata = f["metadata"]

    # theory
    DR = fmetadata.DR1
    p = DR / fmetadata.DI
    @assert fmetadata.DR2 == DR
    l = fmetadata.l1
    @assert fmetadata.l2 == l
    coverm = fmetadata.c1 / fmetadata.m1
    @assert (fmetadata.c2 / fmetadata.m2) == coverm

    dx = fmetadata.L / fmetadata.sN

    K = Ks[iK]
    beta = K * coverm
    Lmax = MinimalModelV2.ksquared_to_L(MinimalModelV2.fr2_km2(beta, l, p, 1.))
    
    xx = identify_peak_gaps.(f["results"][iK].fss, dx)

    xx1 = vcat(getindex.(xx, 1)...)
    xx2 = vcat(getindex.(xx, 2)...);
    
    xx3 = copy(xx2)
    replace!(xx3, 11=>1, 22=>2, 12=>3, 21=>4)

    title = if ismissing(Lmax)
        "Theory predicts no structure"
    else
        @sprintf "Theoretical L_max = %.3g" Lmax
    end
    ax = Axis(place;
        xticks=([1, 2, 3, 4], string.([11, 22, 12, 21])),
        xlabel="Peak gap ID (identities of species on the left and right)",
        ylabel="Peak gap size (length scale)",
        title,
    )
    scatter!(xx3, xx1)
    ax
end
function peak_spacings_rep1(f, iK)
    fig = Figure()
    peak_spacings_rep1!(fig[1,1], f, iK)
    fig
end
function peak_spacings_rep1_full(f)
    Ks = f["Ks"]
    fig = Figure(;
        size=(400, 300 * length(Ks))
    )
    for iK in 1:length(Ks)
        ax = peak_spacings_rep1!(fig[iK,1], f, iK)
        ax.title = (@sprintf "K=%.3g, " Ks[iK]) * ax.title[]
    end
    fig
end
