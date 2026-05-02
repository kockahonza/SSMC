module TwoMMs

using Reexport
@reexport using ..ModifiedMiCRM

using EnumX

Base.@kwdef struct TMMsParams{F}
    # resource supply and dilution
    K::F
    r::F = 1.0
    # upkeep energies
    m1::F
    m2::F
    # leakages
    l1::F
    l2::F
    k1::F = 0.0
    k2::F = 0.0
    # uptake rates
    c1::F
    c2::F
    d1::F = c1
    d2::F = c2
end
function tmmsp_to_mmicrm(tmmps, args...)
    D = fill(0.0, 2, 3, 3)
    D[1, 2, 1] = 1.0 - tmmps.k1
    D[1, 3, 1] = tmmps.k1
    D[2, 3, 1] = 1.0 - tmmps.k2
    D[2, 2, 1] = tmmps.k2
    BMMiCRMParams(
        fill(1.0, 2), fill(1.0, 3),
        [tmmps.m1, tmmps.m2],
        [tmmps.K, 0.0, 0.0],
        fill(tmmps.r, 3),
        [tmmps.l1 0.0 0.0; tmmps.l2 0.0 0.0],
        [tmmps.c1 tmmps.d1 0.0; tmmps.c2 0.0 tmmps.d2],
        D,
        args...
    )
end
export TMMsParams, tmmsp_to_mmicrm

end
