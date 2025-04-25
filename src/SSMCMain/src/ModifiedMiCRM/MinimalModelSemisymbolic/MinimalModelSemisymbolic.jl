module MinimalModelSemisymbolic

using Reexport
@reexport using ..ModifiedMiCRM, ..SpaceMMiCRM

using Polynomials

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
const MinimalModelParams{F} = Union{MinimalModelParamsNoSpace{F},MinimalModelParamsSpace{F}}
export MinimalModelParamsNoSpace, MinimalModelParamsSpace, MinimalModelParams

function add_space(mmp::MinimalModelParamsNoSpace, DN, DG, DR)
    MinimalModelParamsSpace(
        mmp.m, mmp.l, mmp.K, mmp.c, mmp.d,
        DN, DG, DR
    )
end
function get_Ds(mmps::MinimalModelParamsSpace{F}) where {F}
    SA[mmps.DN, mmps.DG, mmps.DR]
end
function mmp_to_mmicrm(mmp::MinimalModelParams)
    MMiCRMParams(
        SA[1.0], SA[1.0, 1.0],
        SA[mmp.m],
        SA[mmp.K, 0.0], SA[1.0, 1.0],
        SA[mmp.l 0.0], SA[mmp.c mmp.d], SArray{Tuple{1,2,2}}(0.0, 1.0, 0.0, 0.0),
    )
end
export add_space, get_Ds, mmp_to_mmicrm

################################################################################
# Dealing with the no space/well mixed case first - finding steady states
################################################################################
include("mmss_nospace.jl")

################################################################################
# Specific fast and hopefully reliable linear stability functions
################################################################################
"""This is hardcoded for a 3 by 3 matrix M1!!"""
function make_K_polynomial_mm(M1, Ds)
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

    Polynomial((t0, t1, t2, t3), :K)
end
export make_K_polynomial_mm

################################################################################
# Main functions, essentially
################################################################################
function analyze_single_mmps_Kpoly(mmps::MinimalModelParamsSpace{F};
    include_extinct=false, threshold=2 * eps(F)
) where {F}
    mmicrm_params = mmp_to_mmicrm(mmps)

    all_steadystates = solve_nospace(mmps; include_extinct, threshold)
    # do the cheap test first, ignore sss with negative values
    physical_steadystates = all_steadystates[nospace_sol_check_physical.(all_steadystates; threshold=2 * threshold)]

    # construct the M1s which are used both for nospace and space linear stability analysis
    M1s_ = make_M1.(Ref(mmicrm_params), physical_steadystates)

    stable_is = nospace_sol_check_stable.(M1s_)
    # stable, physical steady states
    sp_steadystates = physical_steadystates[stable_is]
    M1s = M1s_[stable_is]

    num_nospace_sss = length(sp_steadystates)

    # Spatial linear stability analysis
    Ds = get_Ds(mmps)

    krootss = Vector{F}[]
    num_modes_in_sectionss = Vector{Int}[]
    total_num_modes = 0
    for (ss_i, M1) in enumerate(M1s)
        Kp = make_K_polynomial_mm(M1, Ds)
        kroots = find_ks_that_have_nullspace(Kp; threshold)
        k_samples = sample_ks_from_nullspace_ks(kroots)

        num_modes_in_sections = [find_number_growing_modes(M1, k, Ds; threshold) for k in k_samples]

        push!(krootss, kroots)
        push!(num_modes_in_sectionss, num_modes_in_sections)
        total_num_modes += sum(num_modes_in_sections)
    end

    num_nospace_sss, total_num_modes, sp_steadystates, krootss, num_modes_in_sectionss
end
export analyze_single_mmps_Kpoly

function analyze_single_mmps_kscan(
    mmps::MinimalModelParamsSpace{F}, ks; threshold=2 * eps(F)
) where {F}
    mmicrm_params = mmp_to_mmicrm(mmps)

    all_steadystates = solve_nospace(mmps; include_extinct=false, threshold)
    # do the cheap test first, ignore sss with negative values
    physical_steadystates = all_steadystates[nospace_sol_check_physical.(all_steadystates; threshold=2 * threshold)]

    # construct the M1s which are used both for nospace and space linear stability analysis
    M1s_ = make_M1.(Ref(mmicrm_params), physical_steadystates)

    stable_is = nospace_sol_check_stable.(M1s_)
    # stable, physical steady states
    sp_steadystates = physical_steadystates[stable_is]
    M1s = M1s_[stable_is]

    num_nospace_sss = length(sp_steadystates)

    # Spatial linear stability analysis
    Ds = get_Ds(mmps)

    total_num_modes = 0
    for (ss_i, M1) in enumerate(M1s)

    end

    num_nospace_sss, total_num_modes, sp_steadystates
end
export analyze_single_mmps_kscan

function mm_interactive_k_plot(ks=LinRange(0.0, 100, 10000);
    m_range=LinRange(0.1, 10.0, 1000),
    l_range=LinRange(0.0, 1.0, 100),
    K_range=LinRange(0.1, 100.0, 10000),
    c_range=LinRange(0.1, 100.0, 10000),
    d_range=LinRange(0.1, 100.0, 10000),
    DN_range=vcat(0.0, 10 .^ LinRange(-5, 3, 100)),
    DG_range=vcat(0.0, 10 .^ LinRange(-5, 3, 100)),
    DR_range=vcat(0.0, 10 .^ LinRange(-5, 3, 100)),
    plotimag=false
)
    fig = Figure()

    sg = SliderGrid(fig[1, 1],
        (label="m", range=m_range),
        (label="l", range=l_range),
        (label="K", range=K_range),
        (label="c", range=c_range),
        (label="d", range=d_range),
        (label="DN", range=DN_range),
        (label="DG", range=DG_range),
        (label="DR", range=DR_range),
    )

    sliderobservables = [s.value for s in sg.sliders]
    xx = lift(sliderobservables...) do slvalues...
        mmpf = MinimalModelParamsSpace(slvalues...)
        mmicrm_params = mmp_to_mmicrm(mmpf)

        pssols = find_physical_stable_solutions_nospace(mmpf; include_extinct=true)
        nsols = length(pssols)
        if nsols > 2
            @warn "Found an unexpected number of physical, stable solutions!!"
        end
        nssol = pssols[1]

        Ds = [slvalues[end-2], slvalues[end-1], slvalues[end]]

        do_linstab_for_ks(ks, mmicrm_params, Ds, nssol), nsols
    end
    lambdas = lift(xx -> xx[1], xx)
    nsols = lift(xx -> xx[2], xx)

    mrl = lift(ls -> maximum(real, ls), lambdas)

    fig[1, 2] = label_gl = GridLayout()
    Label(label_gl[1, 1], lift(x -> (@sprintf "mrl = %g" x), mrl))
    Label(label_gl[2, 1], lift(x -> (@sprintf "nsols = %d" x), nsols))

    l1 = lift(ls -> (@view ls[:, 1]), lambdas)
    l2 = lift(ls -> (@view ls[:, 2]), lambdas)
    l3 = lift(ls -> (@view ls[:, 3]), lambdas)

    ax = Axis(fig[2, :])
    for (li, l) in enumerate([l1, l2, l3])
        lines!(ax, ks, lift(real, l);
            color=Cycled(li),
            label=latexstring(@sprintf "\\Re(\\lambda_%d)" li)
        )
        if plotimag
            lines!(ax, ks, lift(imag, l);
                color=Cycled(li),
                linestyle=:dash,
                label=latexstring(@sprintf "\\Im(\\lambda_%d)" li)
            )
        end
    end
    axislegend(ax)
    # on(mrl) do x
    #     @show x
    #     ylims!(ax, (-8 * x, 2 * x))
    # end

    fig, mrl
end
export mm_interactive_k_plot

# NOTE: OLD VERSION
################################################################################
# First deal with solving cubic equation...my god this took a while
################################################################################
function eval_cubic(ps, x)
    ps[1] * x^3 + ps[2] * x^2 + ps[3] * x + ps[4]
end
export eval_cubic

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
export find_real_cubic_roots

# This is the one we actually want
function find_real_positive_cubic_roots(args...; kwargs...)
    real_roots = find_real_cubic_roots(args...; kwargs...)
    typed_zero = zero(eltype(real_roots))
    filter(x -> x > typed_zero, real_roots)
end
export find_real_positive_cubic_roots

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
export make_K_polynomial

end
