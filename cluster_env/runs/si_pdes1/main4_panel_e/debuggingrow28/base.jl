################################################################################
# Row 28 at twice the grid, to tell a real instability from a discretisation one
################################################################################
# Row 28 destabilises at t ~ 8e3 and never settles. At sN=1000 its biomass peaks
# are only 6-7 grid points across at half max, with the solution changing >20% of
# the peak height across a single cell, so the pattern is marginally resolved
# *before* it destabilises - which makes the discretisation the first suspect.
#
# Three runs from one state, each writing its own file:
#   qndf_base   QNDF, the tolerances the production runs used
#   qndf_tight  QNDF, both tolerances 100x tighter
#   fbdf_base   FBDF, a different BDF implementation, base tolerances
#
# qndf_base vs qndf_tight isolates the tolerances, qndf_base vs fbdf_base the
# solver, and all three against ../data/row0028.jld2 isolate the grid. Holding
# the tolerances fixed across the solver pair is deliberate - varying two things
# at once is what made the earlier data/data_st4_all comparison uninterpretable.
#
# BLAS is already pinned to one thread by ../base.jl.
include("../base.jl")

const ROW = 28
const SRC = "../data"     # the production `data` run for this row
const T_START = 10^1.5    # pick the state up here: still smooth and near uniform,
                          # so the resample costs nothing, and well before onset

"""
    double_sN(u)

Resample a `(nu, sN)` field onto `2sN` points: keep every original value and put
the average of each neighbouring pair between them, wrapping at the periodic
edge. That is linear interpolation onto the doubled grid up to a uniform dx/4
shift, which a periodic domain with homogeneous parameters cannot see.

At T_START the field is smooth - neighbours differ by ~1% of the mean - so this
is accurate to O(dx^2) and the two grids really do start from the same state.
"""
function double_sN(u)
    nu, sN = size(u)
    out = similar(u, nu, 2sN)
    for i in 1:sN
        j = mod1(i + 1, sN)
        @views out[:, 2i-1] .= u[:, i]
        @views out[:, 2i] .= (u[:, i] .+ u[:, j]) ./ 2
    end
    out
end

"""
    start_state()

The `data` run's saved state nearest `T_START`, on the doubled grid.
"""
function start_state()
    (; ts, us) = load_pde_states(SRC, ROW)
    k = argmin(abs.(ts .- T_START))
    u0 = double_sN(us[k])
    (; k, t=ts[k], u0, sN=size(u0, 2))
end

"""
    configs()

One entry per run. `name` is also the output filename.
"""
configs() = (
    (; name="qndf_base", solver=QNDF, abstol=1e-7, reltol=1e-9),
    (; name="qndf_tight", solver=QNDF, abstol=1e-9, reltol=1e-11),
    (; name="fbdf_base", solver=FBDF, abstol=1e-7, reltol=1e-9),
)

"""
    main1(; outdir, T, save_step, maxtime, solver_threads)

Run every config from the same doubled state, one thread group each, each writing
its own `<name>.jld2` as soon as it finishes. A config that throws is caught and
reported, so it cannot take the others down with it.

`t` in the output is measured from the pickup, not from the original run's zero -
add `t_start` (~30.5) to line it up with ../data.
"""
function main1(;
    outdir="results",
    T=1e5,                    # past the t ~ 8e3 onset and past where ../data stopped
    save_step=20,
    maxtime=20 * 60 * 60,
    solver_threads=13,        # 3 configs x 13 fills a 40 core node
)
    mkpath(outdir)
    st = start_state()
    sr = load("../systems.jld2", "pde_df")[ROW, :]

    @printf("row %d: picked up save %d at t = %.6g, sN %d -> %d (dx %.5g -> %.5g), L = %.5g\n",
        ROW, st.k, st.t, size(st.u0, 2) ÷ 2, st.sN, sr.L / (st.sN ÷ 2), sr.L / st.sN, sr.L)
    @printf("%d configs, %d solver threads each, T = %.3g, save_step = %d, maxtime = %.3gh\n",
        length(configs()), solver_threads, T, save_step, maxtime / 3600)
    flush(stdout)

    @sync for cfg in configs()
        Threads.@spawn begin
            try
                start = time()
                (; s, saved_ts, saved_us) = run_si_pde(sr.params, sr.Ds, st.u0, T, sr.L;
                    abstol=cfg.abstol, reltol=cfg.reltol, solver=cfg.solver,
                    maxtime, solver_threads, save_step,
                )
                realtime = time() - start
                jldsave(joinpath(outdir, cfg.name * ".jld2");
                    name=cfg.name, solver=string(cfg.solver), abstol=cfg.abstol, reltol=cfg.reltol,
                    row=ROW, t_start=st.t, start_save=st.k, sN=st.sN, L=sr.L, T=T,
                    save_step, solver_threads, maxtime,
                    retcode=s.retcode, final_state=s.u[end], final_T=s.t[end],
                    maxresid=smmicrmmaxresid(s), num_saved=length(saved_ts), realtime,
                    saved_ts, saved_us,
                )
                @printf("%-11s %s at t = %.5g, %d saves, maxresid %.3g, %.2fh\n",
                    cfg.name, string(s.retcode), s.t[end], length(saved_ts),
                    smmicrmmaxresid(s), realtime / 3600)
            catch e
                @printf("%-11s FAILED: %s\n", cfg.name, sprint(showerror, e))
            end
            flush(stdout)
            GC.gc()
        end
    end

    outdir
end
