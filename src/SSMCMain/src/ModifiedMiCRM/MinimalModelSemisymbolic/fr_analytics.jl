function fr_ext_line_K(l, m, c, r=1.0)
    if 0.5 <= l <= 1.0
        4 * l * m * r / c
    elseif 0.0 <= l <= 0.5
        m * r / (c * (1 - l))
    else
        missing
    end
end
function fr_instab_line_K(l, m, c, r=1.0)
    if 0.5 <= l <= 1.0
        m * r / (c * (1 - l))
    else
        missing
    end
end
function fr_cor1_instab_line_K(l, m, c, p, r=1.0)
    if (p / (1 + p)) <= l <= 1.0
        if p < (1 / (2 * (1 - l)))
            l * m * r / (c * p * (1 - l) * (1 - p * (1 - l)))
        else
            missing
        end
    else
        missing
    end
end

function ksquared_to_L(k2; threshold=zero(k2))
    if k2 > threshold
        2pi / sqrt(k2)
    else
        missing
    end
end
ksquared_to_L(k2::Missing; kwargs...) = missing

function fr_lengths_beta_range(l, args...; kwargs...)
    if l > 0.5
        range(4 * l, 1 / (1 - l), args...; kwargs...)
    else
        []
    end
end

function fr_lengths_L0(beta, l, r, D; s=+1)
    chi = 1 - (4 * l) / beta
    if chi < zero(chi)
        missing
    else
        sqrtchi = sqrt(chi)
        k02 = (r / D) * ((4 * l * (chi + s * sqrtchi)) / ((1 - chi) * ((2 * l - 1) - s * sqrtchi)))
        ksquared_to_L(k02)
    end
end

function fr_lengths_Lmax(beta, l, r, D; s=+1)
    chi = 1 - (4 * l) / beta
    if chi < zero(chi)
        missing
    else
        sqrtchi = sqrt(chi)
        km2 = (r / D) * ((2 * l * (2 * l - 1 + s * 2 * (l + 1) * sqrtchi + 3 * chi)) / ((1 - chi) * (2 * l - 1 - s * sqrtchi)))
        ksquared_to_L(km2)
    end
end

################################################################################
# FR analytics v2 where we consider DI=D and DR=p*D with p!=1 !
################################################################################
fr2_beta_lb(l) = l < 0.5 ? 1 / (1 - l) : 4 * l
fr2_beta_ub(l, p) = l / (p * (1 - l) * (1 - p * (1 - l)))

function fr2_instab_beta_range(l, p, n; betamax=1000.0)
    lb = fr2_beta_lb(l)
    ub = fr2_beta_ub(l, p)
    if lb >= ub
        []
    else
        if ub == Inf
            ub = betamax
        end
        range(lb, ub, n)
    end
end

function fr2_km2(beta, l, p, roverD; s=+1)
    chi = 1 - 4 * l / beta
    if chi < 0.0
        return missing
    end
    rootchi = sqrt(chi)
    underroot = (2 * p - 1) * chi + 2 * p * (1 - 2 * (1 - l) * p) * rootchi + (1 - 2 * (1 - l) * p)^2
    if underroot < 0.0
        return missing
    end
    roverD * ((2 * l) / (1 - rootchi)) * ((2 * p * rootchi + s * sqrt(underroot)) / (p * (1 - 2 * (1 - l) * p - rootchi)))
end
