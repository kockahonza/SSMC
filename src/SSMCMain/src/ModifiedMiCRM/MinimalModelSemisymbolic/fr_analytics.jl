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
    # if (1 - 1 / (2 * p)) <= l <= 1.0
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
