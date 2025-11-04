module MinimalModelV2

using Reexport
@reexport using ..ModifiedMiCRM

import ..MinimalModelSemisymbolic

Base.@kwdef struct MMParams{F}
    K::F
    m::F
    c::F
    l::F
    d::F
    k::F = 0.0
    r::F = 1.0
end
export MMParams

function mmp_to_mmicrm(mmp::MMParams{F}; static=true) where {F}
    if static
        SAMMiCRMParams(
            SA[1.0], SA[1.0, 1.0],
            SA[mmp.m],
            SA[mmp.K, 0.0], SA[mmp.r, mmp.r],
            SA[mmp.l mmp.k], SA[mmp.c mmp.d], SArray{Tuple{1,2,2}}(0.0, 1.0, 0.0, 0.0),
        )
    else
        D = fill(0.0, 1, 2, 2)
        D[1, 2, 1] = 1
        BMMiCRMParams(
            [1.0], [1.0, 1.0],
            [mmp.m],
            [mmp.K, 0.0], [mmp.r, mmp.r],
            [mmp.l mmp.k], [mmp.c mmp.d], D,
        )
    end
end
export mmp_to_mmicrm
function mmp_to_smmicrm(mmps::MMParams{F};
    DN=1e-12, DI=1.0, DR=1e-12
) where {F}
    mmicrm_params = mmp_to_mmicrm(mmps)
    SASMMiCRMParams(mmicrm_params, SA[DN, DI, DR])
end
export mmp_to_smmicrm

function mm_get_nospace_sol(mmp::MMParams{F};
    include_extinct=false,
    threshold=100 * eps(F)
) where {F}
    sols = Vector{F}[]

    qa = mmp.c * mmp.d * mmp.m                # qa is always positive
    qb = mmp.m * mmp.r * (mmp.c + mmp.d) - mmp.K * mmp.c * mmp.d * (1 - mmp.l * mmp.k)
    qc = mmp.m * mmp.r^2 - mmp.K * mmp.c * mmp.r * (1.0 - mmp.l)

    Dp1 = qb^2
    Dp2 = 4.0 * qa * qc
    D = Dp1 - Dp2
    if D > threshold # D is positive, two solutions
        sqrtD = sqrt(D)

        N1 = (-qb + sqrtD) / (2.0 * qa)
        if abs(N1) > threshold # only add non-extinct solution
            I1 = mmp.K / (mmp.r + mmp.c * N1)
            R1 = (mmp.c * mmp.l * I1 * N1) / (mmp.r + mmp.d * N1)
            push!(sols, [N1, I1, R1])
        end

        N2 = (-qb - sqrtD) / (2.0 * qa)
        if abs(N2) > threshold # only add non-extinct solution
            I2 = mmp.K / (mmp.r + mmp.c * N2)
            R2 = (mmp.c * mmp.l * I2 * N2) / (mmp.r + mmp.d * N2)
            push!(sols, [N2, I2, R2])
        end
    elseif D > -threshold # then D ~ 0
        if abs(qb) > threshold # only add non-extinct solution
            N = (-qb) / (2.0 * qa)
            I = mmp.K / (mmp.r + mmp.c * N)
            R = (mmp.c * mmp.l * I * N) / (mmp.r + mmp.d * N)
            push!(sols, [N, I, R])
        end
    end

    if include_extinct
        push!(sols, [0.0, mmp.K / mmp.r, 0.0])
    end

    sols
end
export mm_get_nospace_sol

@enumx NospaceSolStability nospace_unstable stable unstable
export NospaceSolStability

function nospacesolstabilities_list()
    @printf "%-20s <-> %d\n" string("doesn't exist") 0
    for i in instances(NospaceSolStability.T)
        @printf "%-20s <-> %d\n" string(i) (Int(i) + 1)
    end
end
export nospacesolstabilities_list

function nospacesolstabilities_to_code(xx)
    vals = sort(Int.(xx) .+ 1; rev=true)
    sum(vals .* (10 .^ (0:(length(xx)-1))); init=0)
end
export nospacesolstabilities_to_code

function analyse_mmp(mmp::MMParams{F};
    DN=1e-12, DI=1.0, DR=1e-12,
    threshold=10 * eps(F),
) where {F}
    # find no space solutions
    nospace_sols = mm_get_nospace_sol(mmp; threshold)

    # get rid of negative solutions - filtering on N is sufficient
    filter!(ss -> ss[1] > threshold, nospace_sols)

    if length(nospace_sols) == 0
        []
    end

    # do linear stability analysis
    mmicrm_params = mmp_to_smmicrm(mmp; DN, DI, DR)
    Ds = SA[DN, DI, DR]

    nospace_sol_stability = []
    for ss in nospace_sols
        M1 = make_M1(mmicrm_params, ss)

        M1evals = eigvals(M1)
        if any(l -> real(l) > threshold, M1evals) # it is unstable even without space
            push!(nospace_sol_stability, NospaceSolStability.nospace_unstable)
            continue
        end

        Kp = MinimalModelSemisymbolic.make_K_polynomial_mm(M1, Ds)
        kroots = find_ks_that_have_nullspace(Kp; threshold)
        k_samples = sample_ks_from_nullspace_ks(kroots)

        has_instability = false
        for k in k_samples
            evals = eigvals!(M1_to_M(M1, Ds, k); sortby=eigen_sortby_reverse)
            if real(evals[1]) > threshold
                has_instability = true
                break
            end
        end

        if has_instability
            push!(nospace_sol_stability, NospaceSolStability.unstable)
        else
            push!(nospace_sol_stability, NospaceSolStability.stable)
        end
    end

    nospace_sol_stability
end
export analyse_mmp

end
