module SymCosmo

using Reexport
@reexport using ..ModifiedMiCRM

Base.@kwdef struct SCParams{F}
    K::F
    m::F
    l::F
    c::F
    d::F
    r::F = 1.0
end
export SCParams

function scp_to_mmicrm(scp::SCParams{F}; static=true) where {F}
    if static
        SAMMiCRMParams(
            SA[1.0, 1.0], SA[1.0, 1.0, 1.0],
            SA[scp.m, scp.m],
            SA[scp.K, 0.0, 0.0], SA[scp.r, scp.r, scp.r],
            SA[scp.l 0.0 0.0; scp.l 0.0 0.0], SA[scp.c scp.d 0.0; scp.c 0.0 scp.d],
            SArray{Tuple{2,3,3}}(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
    else
        D = fill(0.0, 2, 3, 3)
        D[1, 3, 1] = 1.0
        D[2, 2, 1] = 1.0
        BMMiCRMParams(
            [1.0, 1.0], [1.0, 1.0, 1.0],
            [scp.m, scp.m],
            [scp.K, 0.0, 0.0], [scp.r, scp.r, scp.r],
            [scp.l 0.0 0.0; scp.l 0.0 0.0], [scp.c scp.d 0.0; scp.c 0.0 scp.d], D,
        )
    end
end
export scp_to_mmicrm
function scp_to_smmicrm(scps::SCParams{F}, args...;
    DN=1e-12, DI=1.0, DR=1e-12
) where {F}
    mmicrm_params = scp_to_mmicrm(scps)
    SASMMiCRMParams(mmicrm_params, SA[DN, DN, DI, DR, DR], args...)
end
export scp_to_smmicrm

end
