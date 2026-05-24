function do_everything_ps(dir, ps;
    fs_K_step=1,
)
    files = map(ps) do p
        jldopen(joinpath(dir, "p$p.jld2"))
    end;
    Ks = files[1]["metadata"].Ks;

    # Make a df with all data
    full_df = DataFrame(;
        K=Float64[],
        p=Float64[],
        retcode=Any[],
        fT=Any[],
        fs=Any[],
    )
    for (p, f) in zip(ps, files)
        fKs = f["Ks"]
        results = f["results"]
        for i in 1:length(fKs)
            r = results[i]
            for (rc, fT, fs) in zip(r.retcodes, r.fTs, r.fss)
                push!(full_df, (fKs[i], p, rc, fT, fs))
            end
        end
    end
    df = @subset(full_df, :fT .== 1e8, :retcode .== ReturnCode.Success);

    jldsave(joinpath(dir, "df.jld2"); df=full_df, metadata=files[1]["metadata"])

    df.num_surv = num_survivors_in_space.(df.fs, 2);

    # Make plots of final states
    for i in 1:length(files)
        fig = plot_many_final_states(files[i], 5, 1:fs_K_step:length(Ks))
        Label(fig[0,:], (@sprintf "This is for p=%.3g." ps[i]); fontsize=30)
        Makie.save(joinpath(dir, "fs_$i.pdf"), fig)
    end

    # Make outcome plots
    for i in 1:length(files)
        f = files[i]
        fmetadata = f["metadata"]
        Ks = f["Ks"]
        numKs = length(Ks)
        results = f["results"];
        
        surv_threshold = 1e-9
        num_ext = zeros(numKs)
        num_coex = zeros(numKs)
        num_S1 = zeros(numKs)
        num_S2 = zeros(numKs)
        num_bad = zeros(numKs)
        for iK in 1:length(Ks)
            for (rc, fT, fs) in zip(results[iK].retcodes, results[iK].fTs, results[iK].fss)
                if (fT == fmetadata.T) && (rc == ReturnCode.Success)
                    m1 = mean(@view fs[1,:])
                    m2 = mean(@view fs[2,:])
                    if (m1 < surv_threshold) && (m2 < surv_threshold)
                        num_ext[iK] += 1
                    elseif (m1 < surv_threshold)
                        num_S2[iK] += 1
                    elseif (m2 < surv_threshold)
                        num_S1[iK] += 1
                    else
                        num_coex[iK] += 1
                    end
                else
                    num_bad[iK] += 1
                end
            end
        end
        @assert all((num_ext .+ num_coex .+ num_S1 .+ num_S2 .+ num_bad) .== fmetadata.numrepeats)
        
        fig = Figure(;)
        ax = Axis(fig[1,1];
            xscale=log10,
            xlabel=L"\beta=\frac{Kc}{mr},\enspace\text{Normalized energy supply}",
            ylabel=L"\text{Counts}",
        )

        plot_binom_sample!(ax, Ks, num_ext, fmetadata.numrepeats;
            color=PaperColors.extinct1(),
            label="Extinction",
            sl_kwargs=(; markersize=0.),
        )
        plot_binom_sample!(ax, Ks, num_S1, fmetadata.numrepeats;
            color=PaperColors.twomms_1(),
            label="Only 1",
            sl_kwargs=(; markersize=0.),
        )
        plot_binom_sample!(ax, Ks, num_S2, fmetadata.numrepeats;
            color=PaperColors.twomms_2(),
            label="Only 2",
            sl_kwargs=(; markersize=0.),
        )
        plot_binom_sample!(ax, Ks, num_coex, fmetadata.numrepeats;
            color=PaperColors.twomms_coexistence(),
            label="Coexistence",
            sl_kwargs=(; markersize=0.),
        )
        plot_binom_sample!(ax, Ks, num_bad, fmetadata.numrepeats;
            color=:orange,
            label="Bad",
        )
        
        Label(fig[0,:], (@sprintf "This is for p=%.3g." ps[i]);
            fontsize=20,
            tellwidth=false,
        )
        Makie.save(joinpath(dir, "outcomes_$i.pdf"), fig)
    end
end
