################################################################################
# For Fig.3e, multiple systems each solved at 5 different p
################################################################################
include("../base.jl")

"""
    split_systems(infname="systems.jld2", outfnames=("systems1.jld2", "systems2.jld2"))

Split a setup file's `pde_df` in half and write each half out as a setup file of
its own, so the two halves can be run as separate jobs on separate nodes.

Splits on `gdf_row` (the system), not on the row index, so all of a system's ps
land in the same half and each half holds whole systems. With an odd number of
systems the first half gets the extra one.

`run_pde_setup` names its output files by position within the setup file it was
given, so both halves produce a `row0001.jld2` and the two jobs must be pointed
at different outdirs. The original positions are kept in the `orig_rows` key of
each half, which is the only way back to a row's index in the unsplit file;
`gdf_row`/`df_row` still identify the system itself.
"""
function split_systems(infname="systems.jld2",
    outfnames=("systems1.jld2", "systems2.jld2"),
)
    setup = load(infname)
    pde_df = setup["pde_df"]
    metadata = setup["metadata"]

    systems = unique(pde_df.gdf_row)
    nfirst = cld(length(systems), 2)
    halves = (systems[1:nfirst], systems[nfirst+1:end])

    for (outfname, syss) in zip(outfnames, halves)
        orig_rows = findall(in(Set(syss)), pde_df.gdf_row)
        jldsave(outfname; metadata, pde_df=pde_df[orig_rows, :], orig_rows)
        @printf("%s: %d rows, systems %s\n", outfname, length(orig_rows), string(syss))
    end
    flush(stdout)

    outfnames
end

function main1()
    run_pde_setup("systems.jld2", "data";
        save_step=100,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=10 * 60 * 60,
        run_threads=18,
        solver_threads=nothing,
    )
end

# main2/main3 are main1 split over two nodes: half the systems each, 10 rows at a
# time instead of 18, and the node's remaining cores handed to the solvers. 10
# rows in ~60G is 6G a row, above the 5.3G a row that main1_test_DN1e-6 actually
# used; the 18-at-a-time main1 gave each row 3.4G and lost them to
# OutOfMemoryError.
function main2()
    run_pde_setup("systems1.jld2", "data1";
        save_step=100,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=10 * 60 * 60,
        run_threads=10,
        solver_threads=4,
    )
end

function main3()
    run_pde_setup("systems2.jld2", "data2";
        save_step=100,
        abstol=1e-7,
        reltol=1e-9,
        maxtime=10 * 60 * 60,
        run_threads=10,
        solver_threads=4,
    )
end
