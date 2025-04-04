using SSMCMain, SSMCMain.ModifiedMiCRM, SSMCMain.SpaceMMiCRM
using Symbolics
import Nemo

struct MinimalModelParamsNoSpace{F}
    m::F
    l::F
    K::F
    c::F
    d::F
end
struct MinimalModelParamsSpace{F}
    m::F
    l::F
    K::F
    c::F
    d::F
    DN::F
    DG::F
    DR::F
end
function add_space(mmp::MinimalModelParamsNoSpace, DN, DG, DR)
    MinimalModelParamsSpace(
        mmp.m, mmp.l, mmp.K, mmp.c, mmp.d,
        DN, DG, DR
    )
end

function get_Ds(mmps::MinimalModelParamsSpace{F}) where {F}
    SA[mmps.DN, mmps.DG, mmps.DR]
end

const MinimalModelParams{F} = Union{MinimalModelParamsNoSpace{F},MinimalModelParamsSpace{F}}
function mmp_to_mmicrm(mmp::MinimalModelParams)
    MMiCRMParams(
        SA[1.0], SA[1.0, 1.0],
        SA[mmp.m],
        SA[mmp.K, 0.0], SA[1.0, 1.0],
        SA[mmp.l 0.0], SA[mmp.c mmp.d], SArray{Tuple{1,2,2}}(0.0, 1.0, 0.0, 0.0),
    )
end

################################################################################
# Dealing with the no space/well mixed case first - finding steady states
################################################################################
"""
    solve_nospace(mmp::MinimalModelParamsU{F}) where {F}

Solves for the steady states of the minimal model (1 strain, 2 resources).
The function finds all possible steady state solutions by solving the quadratic equation
that results from setting the time derivatives to zero.

Returns a vector of solutions, where each solution is a vector [N, G, R] containing:
- N: strain concentration
- G: glucose (primary resource) concentration
- R: byproduct (secondary resource) concentration

The function always includes the trivial solution [0, K, 0] (extinction state),
and up to two non-trivial solutions if they exist (determined by the discriminant).
"""
function solve_nospace(mmp::MinimalModelParams{F}; include_extinct::Bool=true) where {F}
    m = mmp.m
    l = mmp.l
    K = mmp.K
    c = mmp.c
    d = mmp.d

    sols = Vector{F}[]

    qa = c * d * m
    qb = (c + d) * m - K * c * d
    qc = m - K * c * (1.0 - l)

    Dp1 = qb^2
    Dp2 = 4.0 * qa * qc
    if Dp1 > Dp2
        sqrtD = sqrt(Dp1 - Dp2)

        N1 = (-qb + sqrtD) / (2.0 * qa)
        G1 = K / (1.0 + N1 * c)
        R1 = (G1 * N1 * c * l) / (1.0 + N1 * d)
        push!(sols, [N1, G1, R1])

        N2 = (-qb - sqrtD) / (2.0 * qa)
        G2 = K / (1.0 + N2 * c)
        R2 = (G2 * N2 * c * l) / (1.0 + N2 * d)
        push!(sols, [N2, G2, R2])
    end

    if include_extinct
        push!(sols, [0.0, K, 0.0])
    end

    sols
end

"""Makes sure the returned steady state really is steady"""
function check_solve_nospace(
    mmicrm_params::MMiCRMParams{Ns,Nr,F}, ss;
    threshold=2 * eps(F)
) where {Ns,Nr,F}
    maximum(abs, uninplace(mmicrmfunc!)(ss, mmicrm_params)) < threshold
end
check_solve_nospace(mmp::MinimalModelParams, args...) = check_solve_nospace(mmp_to_mmicrm(mmp), args...)

function nospace_sol_check_physical(ss; threshold=2 * eps(eltype(ss)))
    all(x -> x > -threshold, ss)
end

"""Checks if the returned steady state is stable"""
function nospace_sol_check_stable(M1::AbstractMatrix{F}; threshold=eps(F)) where {F}
    evals = eigvals(M1)
    real(evals[end]) < threshold
end
function nospace_sol_check_stable(
    mmp::MinimalModelParams{F}, ss;
    kwargs...
) where {F}
    nospace_sol_check_stable(make_M1(mmp_to_mmicrm(mmp), ss); kwargs...)
end

function find_physical_stable_solutions_nospace(
    mmp::MinimalModelParams{F};
    include_extinct=false,
    threshold=2 * eps(F),
    physical_threshold=threshold,
    stability_threshold=threshold
) where {F}
    all_sss = solve_nospace(mmp; include_extinct=include_extinct)
    physical_sss = all_sss[nospace_sol_check_physical.(all_sss; threshold=physical_threshold)]
    physical_sss[nospace_sol_check_stable.(Ref(mmp), physical_sss; threshold=stability_threshold)]
end

################################################################################
# First deal with solving cubic equation...my god this took a while
################################################################################
function eval_cubic(ps, x)
    ps[1] * x^3 + ps[2] * x^2 + ps[3] * x + ps[4]
end

function find_real_cubic_roots(a::F, b::F, c::F, d::F;
    threshold=10 * eps(F)
) where {F<:AbstractFloat}
    if abs(a) < threshold
        if abs(b) < threshold
            if abs(c) < threshold # constant polynomial
                if abs(d) < threshold
                    @warn "all coefficients are close to zero"
                end
                return F[]
            else # linear polynomial
                return F[-d/c]
            end
        end # quadratic polynomial
        D = c^2 - 4 * b * d
        if D < -threshold
            return F[]
        elseif abs(D) < threshold
            return F[-c/(2*b)]
        else
            sqrt_disc = sqrt(D)
            return F[-c+sqrt_disc, -c-sqrt_disc] ./ (2 * b)
        end
    else
        return cardano(a, b, c, d; threshold)
    end
end
find_real_cubic_roots(ps::SVector{4}; kwargs...) = find_real_cubic_roots(ps[1], ps[2], ps[3], ps[4]; kwargs...)
function cardano(a::F, b::F, c::F, d::F;
    threshold=10 * eps(F)
) where {F}
    # get rid of a as it could still be very small and we'd need to divide by a^3
    bp = b / a
    cp = c / a
    dp = d / a

    p = (3 * cp - bp^2) / 3
    q = (2 * bp^3 - 9 * bp * cp + 27 * dp) / 27

    D = q^2 / 4 + p^3 / 27
    minushalfq = -q / 2

    if D > threshold # non-zero, positive D - guaranteed 1 real solution
        sqrtD = sqrt(D)
        u1 = minushalfq + sqrtD
        u2 = minushalfq - sqrtD
        t_roots = F[cbrt(u1)+cbrt(u2)]
    else
        Dnot0 = D < -threshold
        if Dnot0 # non-zero, negative D - guaranteed 3 real solutions, but we need complex numbers to get to them
            xx = minushalfq + im * sqrt(-D)
        else # D ~ 0 we get repeated roots
            xx = complex(minushalfq)
        end

        C1 = xx^(1 / 3) # any cube root should work here
        third_angle = exp((2 * pi / 3) * im)
        C2 = C1 * third_angle
        C3 = C2 * third_angle

        t1 = real(C1 - p / (3 * C1))
        t2 = real(C2 - p / (3 * C2))
        t3 = real(C3 - p / (3 * C3))

        if Dnot0 # this is mostly for speed, no need to deal with isapprox when D !~ 0
            t_roots = F[t1, t2, t3]
        else
            eq12 = isapprox(t1, t2; rtol=threshold)
            eq13 = isapprox(t1, t3; rtol=threshold)
            eq23 = isapprox(t2, t3; rtol=threshold)

            if (eq12 + eq13 + eq23) > 1 # all roots are equal (allowing for funky transitivity)
                t_roots = F[t1]
            elseif eq12
                t_roots = F[t1, t3]
            elseif eq13
                t_roots = F[t1, t2]
            elseif eq23
                t_roots = F[t2, t3]
            else
                t_roots = F[t1, t2, t3]
            end
        end
    end

    return t_roots .- (bp / 3)
end

# This is the one we actually want
function find_real_positive_cubic_roots(args...; kwargs...)
    real_roots = find_real_cubic_roots(args...; kwargs...)
    typed_zero = zero(eltype(real_roots))
    filter(x -> x > typed_zero, real_roots)
end

"""This is hardcoded for a 3 by 3 matrix M1!!"""
function make_K_polynomial(M1, Ds)
    a = M1[1, 1]
    b = M1[1, 2]
    c = M1[1, 3]
    d = M1[2, 1]
    e = M1[2, 2]
    f = M1[2, 3]
    g = M1[3, 1]
    h = M1[3, 2]
    i = M1[3, 3]
    o = Ds[1]
    p = Ds[2]
    q = Ds[3]

    t0 = a * e * i - a * f * h - b * d * i + b * f * g + c * d * h - c * e * g
    t1 = -a * e * q - a * i * p + b * d * q + c * g * p - e * i * o + f * h * o
    t2 = a * p * q + e * o * q + i * o * p
    t3 = -o * p * q

    SA[t3, t2, t1, t0]
end

################################################################################
# Main function, essentially
################################################################################
function do_single(mmp::MinimalModelParamsSpace{F};
    include_extinct=false, threshold=eps(F)
) where {F}
    mmicrm_params = mmp_to_mmicrm(mmp)

    all_sss = solve_nospace(mmp; include_extinct)
    # do the cheap test first, ignore sss with negative values
    physical_sss = all_sss[nospace_sol_check_physical.(all_sss; threshold=2 * threshold)]

    # construc the M1s which are used both for nospace and space linear stability analysis
    M1s_all_physical = make_M1.(Ref(mmicrm_params), physical_sss)

    stable_is = nospace_sol_check_stable.(M1s_all_physical)
    # stable, physical steady states
    sp_sss = physical_sss[stable_is]
    M1s = M1s_all_physical[stable_is]

    Kroots = []

    for (ss_i, M1) in enumerate(M1s)
        Kpolynom = make_K_polynomial(M1, SA[mmp.DN, mmp.DG, mmp.DR])
        push!(Kroots, find_real_positive_cubic_roots(Kpolynom; threshold=threshold))
    end

    sp_sss, Kroots
end
