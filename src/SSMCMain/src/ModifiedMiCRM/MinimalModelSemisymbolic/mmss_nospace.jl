@enumx MMNoSpaceSolType imaginary nonpositive onepositive bothpositive
export MMNoSpaceSolType

"""
This assumes that it always the + root of the two nospace solutions that
is stable. This has been somewhat tested numerically and Jaime sais it should
be clear from the bifurcation diagram.
"""
function mmp_nospace_sol(m, l, K, c, d; threshold=2 * eps(m))
    qa = c * d * m                # qa is always positive
    qb = (c + d) * m - K * c * d  # qb can be either
    qc = m - K * c * (1.0 - l)    # qc as well

    Dp1 = qb^2
    Dp2 = 4.0 * qa * qc
    D = Dp1 - Dp2

    if D < -threshold # D is negative, only extinct solution
        return MMNoSpaceSolType.imaginary, nothing
    elseif D < threshold # D ~ 0
        sqrtD = 0.0
    else
        sqrtD = sqrt(D)
    end
    Np = (-qb + sqrtD) / (2.0 * qa) # Np is guaranteed > Nm
    Nm = (-qb - sqrtD) / (2.0 * qa)

    if Np < threshold
        return MMNoSpaceSolType.nonpositive, nothing
    else
        Gp = K / (1.0 + Np * c)
        Rp = (Gp * Np * c * l) / (1.0 + Np * d)
        sol = [Np, Gp, Rp]
        if Nm < threshold
            return MMNoSpaceSolType.onepositive, sol
        else
            Gm = K / (1.0 + Nm * c)
            Rm = (Gm * Nm * c * l) / (1.0 + Nm * d)
            return MMNoSpaceSolType.bothpositive, (sol, [Nm, Gm, Rm])
        end
    end
end
mmp_nospace_sol(mmp::MinimalModelParams; kwargs...) = mmp_nospace_sol(mmp.m, mmp.l, mmp.K, mmp.c, mmp.d; kwargs...)
export mmp_nospace_sol

################################################################################
# Old, perhaps too general method
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
function solve_nospace(
    mmp::MinimalModelParams{F};
    include_extinct::Bool=true, threshold=eps(F)
) where {F}
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
    D = Dp1 - Dp2
    if D > threshold # D is positive, two solutions
        sqrtD = sqrt(D)

        N1 = (-qb + sqrtD) / (2.0 * qa)
        if abs(N1) > threshold # only add non-extinct solution
            G1 = K / (1.0 + N1 * c)
            R1 = (G1 * N1 * c * l) / (1.0 + N1 * d)
            push!(sols, [N1, G1, R1])
        end

        N2 = (-qb - sqrtD) / (2.0 * qa)
        if abs(N2) > threshold # only add non-extinct solution
            G2 = K / (1.0 + N2 * c)
            R2 = (G2 * N2 * c * l) / (1.0 + N2 * d)
            push!(sols, [N2, G2, R2])
        end
    elseif D > -threshold # then D ~ 0
        if abs(qb) > threshold # only add non-extinct solution
            N = (-qb) / (2.0 * qa)
            G = K / (1.0 + N * c)
            R = (G * N * c * l) / (1.0 + N * d)
            push!(sols, [N, G, R])
        end
    end

    if include_extinct
        push!(sols, [0.0, K, 0.0])
    end

    sols
end
export solve_nospace

"""Makes sure the returned steady state really is steady"""
function check_solve_nospace(
    mmicrm_params::AbstractMMiCRMParams{F}, ss;
    threshold=2 * eps(F)
) where {F}
    maximum(abs, uninplace(mmicrmfunc!)(ss, mmicrm_params)) < threshold
end
check_solve_nospace(mmp::MinimalModelParams, args...) = check_solve_nospace(mmp_to_mmicrm(mmp), args...)
export check_solve_nospace

function nospace_sol_check_physical(ss; threshold=2 * eps(eltype(ss)))
    all(x -> isfinite(x) && (x > -threshold), ss)
end
export nospace_sol_check_physical

"""Checks if the returned steady state is stable"""
function nospace_sol_check_stable(M1::AbstractMatrix{F}; threshold=eps(F)) where {F}
    evals = eigvals(M1)
    maximum(real, evals) < threshold
end
function nospace_sol_check_stable(
    mmp::MinimalModelParams{F}, ss;
    kwargs...
) where {F}
    nospace_sol_check_stable(make_M1(mmp_to_mmicrm(mmp), ss); kwargs...)
end
export nospace_sol_check_stable

function find_physical_stable_solutions_nospace(
    mmp::MinimalModelParams{F};
    include_extinct=false,
    threshold=2 * eps(F),
    physical_threshold=threshold,
    stability_threshold=threshold
) where {F}
    all_sss = solve_nospace(mmp; include_extinct, threshold)
    physical_sss = all_sss[nospace_sol_check_physical.(all_sss; threshold=physical_threshold)]
    physical_sss[nospace_sol_check_stable.(Ref(mmp), physical_sss; threshold=stability_threshold)]
end
export find_physical_stable_solutions_nospace
