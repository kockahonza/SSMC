using Peaks

function identify_peaks(fs)
    pks1 = peakproms(findmaxima(fs[1,:]); min=0.1)
    pks2 = peakproms(findmaxima(fs[2,:]); min=0.1);
    pks_is = [pks1.indices; pks2.indices];
    pks_ids = [fill(1, length(pks1.indices)); fill(2, length(pks2.indices))];

    sp = sortperm(pks_is)

    pks_is[sp], pks_ids[sp]
end
function identify_peak_gaps(fs, dx=nothing)
    if isnothing(dx)
        dx = 1.
        @warn "Did not provide dx to identify_peak_gaps using dx=1."
    end
    pks_is, pks_ids = identify_peaks(fs)
    if isempty(pks_is)
        return (Float64[], Int[])
    end
    sN = size(fs, 2)

    gaps = Float64[]
    gap_ids = Int[]
    for i in 1:(length(pks_is) - 1)
        push!(gaps, (pks_is[i+1] - pks_is[i]) * dx)
        push!(gap_ids, pks_ids[i] * 10 + pks_ids[i+1])
    end
    push!(gaps, (pks_is[1] + sN - pks_is[end]) * dx)
    push!(gap_ids, pks_ids[end] * 10 + pks_ids[1])
    gaps, gap_ids
end
