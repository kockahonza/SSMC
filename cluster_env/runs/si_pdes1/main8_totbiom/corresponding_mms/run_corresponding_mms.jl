################################################################################
# corresponding_mms - the minimal model PDE runs matching si_pdes1/main8_totbiom
#
# For every SI system in ../systems.jld2, fit the MM d (m=c=1, K and l the SI
# systems' own) so the MM disprel peak height matches the SI one, as in
# notebooks/paper_figures/Fitting MM to SI disprels.ipynb (fit_mm_d from
# si_disprel2, kweight=0, ie height only). Then solve the MM PDE from its
# analytic well-mixed steady state at every p main8 ran, with the
# mm_pdes_vs_p machinery.
#
# The fit depends on p (it is the DR the disprel is taken at), and main8 has
# two. So d is fitted once per (system, fit_p) for each p in main8, and each of
# those MMs is run at every p - rows with fit_p == p are the "fit at the p you
# run" ones, the others show what a single per-system d does across p.
#
# Everything the SI PDEs fix is reused: DN, DI, DR=p, sN, T, and the domain L of
# the SI run at that (system, p), so an MM row and its SI row are on the same
# domain. The u0 is the MM hss perturbed proportionally by the same epsilon the
# SI u0s got (perturb_u0_uniform_prop).
#
# Everything this writes stays in this directory: fits.jld2, setup.jld2 and
# main1/. Run from here.
################################################################################
include("../../../mm_pdes_vs_p/base.jl")
include("../../../si_disprel2/base.jl") # for fit_mm_d

const SI_SETUP = "../systems.jld2"

"""
    make_fits_and_setup(; kwargs...)

Fit d for every (system, fit_p) into `fits.jld2` and write `setup.jld2`, the
`run_pde_setup` plan with one row per (system, fit_p, p). Refuses to overwrite
either file.
"""
function make_fits_and_setup(;
    fits_fname="fits.jld2",
    setup_fname="setup.jld2",
    fit_ks=10 .^ range(-5, 5.0, 1000), # as in the fitting notebook
    peak_ks=10 .^ range(-5, 5.0, 20001),
    seed=20261012,
)
    (isfile(fits_fname) || isfile(setup_fname)) && error("$fits_fname or $setup_fname already exists")

    si = load(SI_SETUP)
    simd, si_df = si["metadata"], si["pde_df"]
    (; K, l, DN, DI, ps, epsilon) = simd

    # one fit per (system, fit_p), from the SI row at that p
    fits_df = DataFrame(map(eachrow(si_df)) do r
        f = fit_mm_d(r.params, r.ode_hss, r.Ds, K, l, r.p, fit_ks, 0.0; DN, DI)
        isnothing(f) && error("SI sys_i=$(r.sys_i) p=$(r.p) is not spatially unstable")
        (; r.sys_i, r.df_row, fit_p=r.p, f.fit_d, f.si_k, f.mm_k, f.si_h, f.mm_h, f.si_mrls, f.mmls)
    end)
    jldsave(fits_fname; fits_df, fit_ks, K, l, DN, DI, si_setup=SI_SETUP)

    @printf("%6s %6s %12s %12s %12s %12s %12s\n", "sys_i", "fit_p", "fit_d", "si_h", "mm_h", "si_k", "mm_k")
    for f in eachrow(fits_df)
        @printf("%6d %6g %12.5g %12.5g %12.5g %12.5g %12.5g\n", f.sys_i, f.fit_p, f.fit_d, f.si_h, f.mm_h, f.si_k, f.mm_k)
    end

    Random.seed!(seed)
    rows = NamedTuple[]
    for f in eachrow(fits_df)
        mmp = MMParams(; K, l, m=1.0, c=1.0, d=f.fit_d)
        hss = mmv3_get_hss_unique(mmp)
        mmicrm_params = mmp_to_mmicrm(mmp; static=false)
        # one u0 per fitted MM, shared across the ps it runs at, like the SI u0s
        sN = only(unique(si_df.sN))
        u0 = perturb_u0_uniform_prop(1, 2, expand_u0_to_size((sN,), hss), epsilon)
        for (p_i, p) in enumerate(ps)
            sir = only(eachrow(si_df[(si_df.sys_i.==f.sys_i).&(si_df.p.==p), :]))
            ls = mm_peak_k(mmp, DN, DI, p, hss[1], peak_ks)
            push!(rows, (;
                p_i, p, rep=1,
                f.sys_i, f.df_row, f.fit_p, f.fit_d,
                mmp, mmicrm_params, Ds=[DN, DI, p],
                hss, peak_k=ls.peak_k, peak_h=ls.peak_h, mm_unstable=ls.unstable, peak_at_edge=ls.at_edge,
                si_pde_df_row=rownumber(sir),
                L=sir.L, sN, T=sir.T, u0,
            ))
        end
    end
    pde_df = DataFrame(rows)

    metadata = (; K, l, DN, DI, ps, epsilon, seed, peak_ks, si_setup=SI_SETUP, si_metadata=simd)
    jldsave(setup_fname; metadata, pde_df)

    @printf("\n%6s %6s %6s %12s %12s %10s %8s  %s\n", "row", "sys_i", "fit_p", "p", "peak_h", "L", "npeaks", "linstab")
    for (i, r) in enumerate(eachrow(pde_df))
        @printf("%6d %6d %6g %12g %12.5g %10.4g %8.3g  %s\n", i, r.sys_i, r.fit_p, r.p, r.peak_h, r.L,
            num_peaks_in(r.L, r.peak_k),
            r.mm_unstable ? (r.peak_at_edge ? "unstable (peak at ks edge!)" : "unstable") : "STABLE")
    end
    @printf("wrote %s and %s (%d rows, %d MM-stable)\n", fits_fname, setup_fname, nrow(pde_df), count(!, pde_df.mm_unstable))
    flush(stdout)

    pde_df
end

"""
    main1()

All rows of `setup.jld2` into `main1/`, with main8's solver settings.
"""
function main1()
    run_pde_setup("setup.jld2", "main1";
        save_step=20,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=3 * 60 * 60,
        run_threads=nthreads(),
        solver_threads=nothing,
    )
end
