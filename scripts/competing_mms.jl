using SSMCMain, SSMCMain.ModifiedMiCRM

using Base.Threads
using OhMyThreads

function make_comp_mms_params(
    K,
    m1, m2,
    l1, l2,
    c1, c2,
    args...;
    d1=1.0, k1=0.0,
    d2=1.0, k2=0.0,
    r=1.0,
)
    D = fill(0.0, 2, 3, 3)
    D[1, 2, 1] = 1.0
    D[2, 3, 1] = 1.0
    BMMiCRMParams(
        fill(1.0, 2), fill(1.0, 3),
        [m1, m2],
        [K, 0.0, 0.0],
        fill(r, 3),
        [l1 k1 0.0; l2 0.0 k2],
        [c1 d1 0.0; c2 0.0 d2], D,
        args...
    )
end
