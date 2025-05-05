module MinimalModelSemisymbolic

using Reexport
@reexport using ..ModifiedMiCRM

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
    SAMMiCRMParams(
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

        num_modes_in_sections = zeros(length(k_samples))
        for (i, k) in enumerate(k_samples)
            evals = eigvals!(M1_to_M(M1, Ds, k))
            num_modes_in_sections[i] = count(x -> real(x) > threshold, evals)
        end

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

    if num_nospace_sss > 0
        # Spatial linear stability analysis
        Ds = get_Ds(mmps)

        M_tmp = copy(M1s[1])
        ss_stability = Vector{Bool}(undef, length(M1s))
        for (ss_i, M1) in enumerate(M1s)
            stable = true
            for k in ks
                M_tmp .= M1
                for i in 1:length(Ds)
                    M_tmp[i, i] -= k^2 * Ds[i]
                end
                evals = eigvals(M_tmp)
                if any(l -> real(l) > threshold, evals)
                    stable = false
                    break
                end
            end
            ss_stability[ss_i] = stable
        end
        num_unstable_sss = count(!, ss_stability)
        num_nospace_sss, num_unstable_sss, sp_steadystates, ss_stability
    else
        num_nospace_sss, 0, sp_steadystates, Bool[]
    end
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

include("old_polynomial_stuff.jl")

end
