# Reduce the weird rows to what the plots need: total biomass over (t, x) with a
# thinned time axis, plus the small per-row scalars and the final state. The full
# saved_us series is ~14G over these 12 runs and the link off dias runs at well
# under 1 MB/s, so it stays here.
const WEIRD = [9, 15, 18, 21, 31, 50]
const MAXNT = 1200

setup_df = load("systems.jld2", "pde_df")
Ns, Nr = get_Ns(setup_df.params[1])

out = Dict{String,Any}()
for (name, dir) in ("data" => "data", "st4" => "data_st4_all")
    d = collect_pde_runs(dir)
    for i in WEIRD
        k = findfirst(==(i), d.pde_df_row)
        isnothing(k) && continue
        r = d[k, :]
        (; ts, us) = load_pde_states(dir, i)
        keep = findall(>(0.0), ts)
        keep = keep[round.(Int, range(1, length(keep), min(length(keep), MAXNT)))]
        ntot = Float32.(permutedims(reduce(hcat, [vec(sum(view(u, 1:Ns, :), dims=1)) for u in us[keep]])))
        out["$(name)_$(i)"] = (; ts=ts[keep], ntot,
            final_state=r.final_state, retcode=string(r.retcode), final_T=r.final_T,
            num_saved=r.num_saved, maxresid=r.maxresid, realtime=r.realtime,
            gdf_row=r.gdf_row, p=r.p, L=r.L, sN=r.sN)
        @printf("%s row %4d: %d saved -> %d kept, ntot %s\n", name, i, length(ts), length(keep), string(size(ntot)))
        flush(stdout)
        us = nothing
        GC.gc()
    end
end

jldsave("weird_rows.jld2"; runs=out, weird=WEIRD, maxnt=MAXNT)
@printf("wrote weird_rows.jld2, %.1f MB\n", filesize("weird_rows.jld2") / 1e6)
