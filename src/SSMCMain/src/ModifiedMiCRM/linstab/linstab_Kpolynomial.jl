################################################################################
# Utils bits
################################################################################
function filter_real_positive(
    vals::AbstractVector{F};
    threshold=2 * eps(F),
    include_zeros=false
) where {F}
    real_positive_vals = F[]
    for val in vals
        if val > threshold
            push!(real_positive_vals, val)
        elseif include_zeros && (val > -threshold)
            push!(real_positive_vals, 0.0)
        end
    end
    real_positive_vals
end
function filter_real_positive(
    vals::AbstractVector{Complex{F}};
    threshold=2 * eps(F),
    include_zeros=false
) where {F}
    real_positive_vals = F[]
    for val in vals
        if abs(imag(val)) < threshold
            rr = real(val)
            if rr > threshold
                push!(real_positive_vals, rr)
            elseif include_zeros && (rr > -threshold)
                push!(real_positive_vals, 0.0)
            end
        end
    end
    real_positive_vals
end
export filter_real_positive

function sort_and_remove_duplicates!(vals; threshold=2 * eps(eltype(vals)))
    sort!(vals)

    if length(vals) >= 2
        i = 1
        while true
            if i >= length(vals)
                break
            elseif (vals[i+1] - vals[i]) < threshold
                vals[i] = (vals[i] + vals[i+1]) / 2
                deleteat!(vals, i + 1)
            else
                i += 1
            end
        end
    end
end
export sort_and_remove_duplicates!

################################################################################
# Making the K polynomial
################################################################################
function make_K_polynomial(M1, Ds)
    throw(ErrorException("Not implemented"))
end
export make_K_polynomial

################################################################################
# Using the polynomial
################################################################################
function find_ks_that_have_nullspace(
    Kpoly::Polynomial;
    threshold=2 * eps(eltype(Kpoly))
)
    rs = roots(Kpoly)

    rs = filter_real_positive(rs; threshold)
    sort_and_remove_duplicates!(rs; threshold)

    sqrt.(rs)
end
export find_ks_that_have_nullspace

function sample_ks_from_nullspace_ks(nks::AbstractVector{F}) where {F}
    if length(nks) == 0
        k_samples = F[1.0]
    elseif length(nks) == 1
        k_samples = F[nks[1]/2, 2*nks[1]]
    else
        k_samples = F[nks[1]/2]

        for i in 2:(length(nks))
            push!(k_samples, (nks[i-1] + nks[i]) / 2)
        end

        push!(k_samples, 2 * nks[end])
    end
    k_samples
end
export sample_ks_from_nullspace_ks

function find_sample_ks(args...; kwargs...)
    sample_ks_from_nullspace_ks(find_ks_that_have_nullspace(args...; kwargs...))
end
export find_sample_ks
