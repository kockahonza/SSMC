module MinimalModelV3

using Reexport
@reexport using ..ModifiedMiCRM
@reexport using ..MinimalModelV2

################################################################################
# Finding the homogeneous steady state through analytical expressions
################################################################################

function mmv3_get_hss_extinct(mmp::MMParams{F}) where {F}
    F[zero(F), mmp.K/mmp.r, zero(F)]
end

"""
    quadratic_roots(a, b, c) -> (x1, x2)

Real roots of `a*x^2 + b*x + c`, ascending, assuming `a != 0`. Returns
`(NaN, NaN)` when the roots are complex; since `NaN > 0` is `false`, callers
filtering for positive roots need no separate check.

`b == 0` and `c == 0` are handled by the general formulas, no branch needed.
"""
function quadratic_roots(a::F, b::F, c::F) where {F}
    disc = b^2 - 4 * a * c
    disc < 0 && return (F(NaN), F(NaN))
    sq = sqrt(disc)

    # Whichever of -b±sq adds two like-signed terms is computed without
    # cancellation; the other root then comes from x1*x2 == c/a.
    far = b >= 0 ? (-b - sq) / (2 * a) : (-b + sq) / (2 * a)
    near = iszero(far) ? far : (c / a) / far

    minmax(near, far)
end

"""
Finds the non-extinct
"""
function mmv3_get_hss_quadratic(mmp::MMParams{F}) where {F}
    @assert mmp.k == zero(F)

    qa = mmp.m * mmp.c * mmp.d
    qb = mmp.m * mmp.r * (mmp.c + mmp.d) - mmp.K * mmp.c * mmp.d
    qc = mmp.m * mmp.r^2 - mmp.K * mmp.c * mmp.r * (1 - mmp.l)

    sols = Vector{F}[]
    for N in quadratic_roots(qa, qb, qc)
        N > zero(F) || continue
        I = mmp.K / (mmp.r + mmp.c * N)
        R = (mmp.c * mmp.l * I * N) / (mmp.r + mmp.d * N)
        push!(sols, [N, I, R])
    end
    sols
end

function mmv3_get_hss_all(mmp::MMParams)
    sols = mmv3_get_hss_quadratic(mmp)
    push!(sols, mmv3_get_hss_extinct(mmp))
    sols
end
export mmv3_get_hss_all

function mmv3_get_hss_stable(mmp::MMParams{F}; rtol=100 * eps(F)) where {F}
    sols = mmv3_get_hss_all(mmp)
    mmicrm_params = mmp_to_mmicrm(mmp; static=false)

    M1 = Matrix{F}(undef, sum(get_Ns(mmicrm_params)), sum(get_Ns(mmicrm_params)))
    filter!(sols) do ss
        make_M1!(M1, mmicrm_params, ss)
        tol = rtol * opnorm(M1, 1) # fancy metric for testing near 0
        evals = eigvals!(M1)
        all(l -> real(l) < tol, evals)
    end
end
export mmv3_get_hss_stable

@enumx MMV3SolChoice Max Min
function mmv3_get_hss_unique(mmp::MMParams{F}, solchoice=MMV3SolChoice.Max; rtol=100 * eps(F)) where F
    sols = mmv3_get_hss_stable(mmp; rtol)
    if solchoice == MMV3SolChoice.Max
        _, i = findmax(s->s[1], sols)
    elseif solchoice == MMV3SolChoice.Min
        _, i = findmin(s->s[1], sols)
    else
        throw(ArgumentError("Unrecognized solchoice"))
    end
    sols[i]
end
export mmv3_get_hss_unique, MMV3SolChoice

################################################################################
# FR analytics 1 where we get the instability region with some heurestics
################################################################################
"""A simplified extline assuming some conditions"""
function fr3_beta1(l, gamma)
    b1 = (1 + gamma) / gamma
    b2 = 1 / (1 - l)
    pos_b = min(b1, b2)

    rootbit = l * (l + gamma - 1)
    if rootbit < 0.
        pos_b
    else
        real_bplu = (2 * l + gamma - 1 + 2 * sqrt(rootbit)) / gamma
        max(pos_b, real_bplu)
    end
end
export fr3_beta1

# Extras for debugging
function fr3_beta1_complex(l, gamma)
    b1 = (1 + gamma) / gamma
    b2 = 1 / (1 - l)
    pos_b = min(b1, b2)
    @show b1 == pos_b

    rootbit = l * (l + gamma - 1)
    if rootbit < 0.
        @printf "Seeing rootbit < 0\n"
        return pos_b
    else
        real_bmin = (2 * l + gamma - 1 - 2 * sqrt(rootbit)) / gamma
        real_bplu = (2 * l + gamma - 1 + 2 * sqrt(rootbit)) / gamma
        @show real_bmin pos_b

        if pos_b < real_bmin
            @warn "Getting stuff that uses beta < beta_minus!!!"
            return pos_b
        else
            return max(pos_b, real_bplu)
        end
    end
end
export fr3_beta1_complex

################################################################################
# FR dispersion relation
################################################################################
function fr3_disprel_simple(mmp::MMParams, DN, DI, DR, Nss, ks)
    u = mmp.r + mmp.c * Nss
    v = mmp.r + mmp.d * Nss
    lIsq = DI / u
    lRsq = DR / v
    A = Nss * mmp.d * mmp.c * mmp.l * mmp.K * mmp.r / (u * v^2)
    I1 = Nss * mmp.c^2 * (1 - mmp.l) * mmp.K / (u^2)
    I2 = Nss^2 * mmp.d * mmp.c^2 * mmp.l * mmp.K / (u^2 * v)
    ksqs = ks .^ 2
    denom1 = (1 .+ lRsq .* ksqs)
    denom2 = (1 .+ lIsq .* ksqs)
    ((-DN) .* ksqs) .+ (A ./ denom1) .- (I1 ./ denom2) .- (I2 ./ (denom1 .* denom2))
end
export fr3_disprel_simple


end
