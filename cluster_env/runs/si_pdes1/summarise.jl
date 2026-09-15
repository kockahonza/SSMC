using SSMCMain
using JLD2, Printf

med(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n+1)÷2] : 0.5*(s[n÷2]+s[n÷2+1]))

outdir = ARGS[1]
fs = sort(filter(f -> startswith(f, "row") && endswith(f, ".jld2"), readdir(outdir)))
@printf("%4s %10s %8s %5s %9s %16s %10s %10s %6s %9s %9s %10s %10s %7s\n",
    "row", "p", "L", "sN", "T", "retcode", "final_T", "maxresid", "nsav",
    "ts[2]", "dt_med", "min_u", "max_u", "wall_h")
for f in fs
    jldopen(joinpath(outdir, f)) do d
        ts = d["saved_ts"]; u = d["final_state"]
        dts = length(ts) > 2 ? diff(ts) : [NaN]
        @printf("%4d %10.4g %8.4g %5d %9.3g %16s %10.4g %10.2e %6d %9.3g %9.3g %10.2e %10.4g %7.2f\n",
            d["pde_df_row"], d["p"], d["L"], d["sN"], d["T"], string(d["retcode"]),
            d["final_T"], d["maxresid"], d["num_saved"],
            length(ts) > 1 ? ts[2] : NaN, med(dts), minimum(u), maximum(u),
            d["realtime"]/3600)
    end
end

if length(ARGS) > 1 && isfile(ARGS[2])
    println("\n--- setup pde_df ---")
    s = load(ARGS[2])
    pdf = s["pde_df"]
    println(names(pdf))
    for r in eachrow(pdf)
        @printf("gdf_row=%s df_row=%s p=%.4g L=%.5g sN=%d T=%.3g u0 size=%s\n",
            string(r.gdf_row), string(r.df_row), r.p, r.L, r.sN, r.T, string(size(r.u0)))
    end
    println("\nsetup metadata:"); println(s["metadata"])
end
