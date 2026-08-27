# Every key each row file holds except `saved_us`, the coarsening state series -
# which is all but a few MB of a row file, and 35G over the two runs - plus a
# `biomass_t` reduced out of it. `saved_ts` stays, so the step pacing is still
# readable, and so does `final_state`.
#
# One DataFrame over both runs with a `run` column, since the row files number
# themselves by position in systems.jld2 and so mean the same thing in each.
#
# biomass_t[k] is the spatial *mean* over x of the total strain density at
# saved_ts[k], ie mean(sum(u[1:Ns, :], dims=1)). That is the quantity directly
# comparable to the well-mixed steady state's sum over strains; multiply by L for
# the domain integral. saved_us is loaded a row at a time and dropped.
const RUNS = ("data" => "data", "st4" => "data_st4_all")
const DROP = Set(["saved_us", "settings"])

function row_stats(dir, Ns)
    fnames = sort(filter(f -> startswith(f, "row") && endswith(f, ".jld2"), readdir(dir)))
    rows = map(fnames) do f
        r = jldopen(joinpath(dir, f)) do d
            scalars = (; (Symbol(k) => d[k] for k in filter(!in(DROP), keys(d)))...)
            us = d["saved_us"]
            biomass_t = [sum(view(u, 1:Ns, :)) / size(u, 2) for u in us]
            merge(scalars, (; biomass_t))
        end
        GC.gc()   # a single row's state series can be 1.7G
        r
    end
    sort!(DataFrame(rows), :pde_df_row)
end

setup_df = load("systems.jld2", "pde_df")
Ns, Nr = get_Ns(setup_df.params[1])

parts = DataFrame[]
settings = Dict{String,Any}()
for (nm, dir) in RUNS
    df = row_stats(dir, Ns)
    insertcols!(df, 1, :run => fill(nm, nrow(df)))
    push!(parts, df)
    settings[nm] = jldopen(joinpath(dir, "settings.jld2")) do d
        d["settings"]
    end
    @printf("%-4s %-13s %d rows, %d time points, biomass_t lengths match saved_ts: %s\n",
        nm, dir, nrow(df), sum(length, df.saved_ts),
        all(length(a) == length(b) for (a, b) in zip(df.saved_ts, df.biomass_t)))
    flush(stdout)
end

df = vcat(parts...)
jldsave("run_stats.jld2"; df, settings, Ns, Nr)

@printf("\ncolumns: %s\n", string(names(df)))
@printf("wrote run_stats.jld2, %.1f MB for %d rows\n", filesize("run_stats.jld2") / 1e6, nrow(df))
