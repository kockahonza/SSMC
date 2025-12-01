using SSMCMain, SSMCMain.ModifiedMiCRM, SSMCMain.ModifiedMiCRM.MinimalModelV2

using Base.Threads
using OhMyThreads
using ProgressMeter

Base.@kwdef struct CMMsParams{F}
    K::F
    m1::F
    c1::F
    l1::F
    m2::F
    c2::F
    l2::F
    d1::F = c1
    d2::F = c2
    k1::F = 0.0
    k2::F = 0.0
    r::F = 1.0
end
function cmmsp_to_mmicrm(cmmsp, args...)
    D = fill(0.0, 2, 3, 3)
    D[1, 2, 1] = 1.0
    D[2, 3, 1] = 1.0
    BMMiCRMParams(
        fill(1.0, 2), fill(1.0, 3),
        [cmmsp.m1, cmmsp.m2],
        [cmmsp.K, 0.0, 0.0],
        fill(cmmsp.r, 3),
        [cmmsp.l1 cmmsp.k1 0.0; cmmsp.l2 0.0 cmmsp.k2],
        [cmmsp.c1 cmmsp.d1 0.0; cmmsp.c2 0.0 cmmsp.d2], D,
        args...
    )
end

function cmmsp_get_single_mm_results(cmmsp::CMMsParams, DN1, DN2, DI, DR1, DR2)
    mmp1 = MMParams(;
        K=cmmsp.K,
        m=cmmsp.m1, c=cmmsp.c1, l=cmmsp.l1,
        d=cmmsp.d1, k=cmmsp.k1, r=cmmsp.r,
    )
    mmr1 = get_simplified_analysis(mmp1; DN=DN1, DI=DI, DR=DR1)
    mmp2 = MMParams(;
        K=cmmsp.K,
        m=cmmsp.m2, c=cmmsp.c2, l=cmmsp.l2,
        d=cmmsp.d2, k=cmmsp.k2, r=cmmsp.r,
    )
    mmr2 = get_simplified_analysis(mmp2; DN=DN2, DI=DI, DR=DR2)
    (mmr1, mmr2)
end

################################################################################
# Running repeated spatial sims
################################################################################
@enumx CMMsSpatialOutcome Extinction Coexistence S1Only S2Only
function cmms_spatial_sol_to_outcome(s; extthreshold=1e-6)
    s1surv = mean(s.u[end][1, :]) > extthreshold
    s2surv = mean(s.u[end][2, :]) > extthreshold
    if s1surv && s2surv
        CMMsSpatialOutcome.Coexistence
    elseif s1surv
        CMMsSpatialOutcome.S1Only
    elseif s2surv
        CMMsSpatialOutcome.S2Only
    else
        CMMsSpatialOutcome.Extinction
    end
end

function run_siny_cmms(
    cmms_mmicrcm_params::AbstractMMiCRMParams, numrepeats, T,
    L, sN,
    meanN0, numwaves, waveampfactor,
    ;
    Ds=SA[1e-6, 1e-6, 1.0, 1.0, 1.0],
    tol=1e-8,
    extthreshold=100 * tol,
    maxtime=60,
    save_sols=true,
    # threading
    run_threads=1,
    total_threads=nthreads(),
    solver_threads=div(total_threads, run_threads),
)
    dx = L / sN

    sols = Vector{Any}(undef, numrepeats)
    retcodes = Vector{ReturnCode.T}(undef, numrepeats)
    fss = Vector{Matrix{Float64}}(undef, numrepeats)
    fTs = Vector{Float64}(undef, numrepeats)
    outcomes = Vector{CMMsSpatialOutcome.T}(undef, numrepeats)

    prog = Progress(numrepeats)
    @tasks for i in 1:numrepeats
        @set ntasks = run_threads
        u0 = get_2mm_u0_sines(sN, dx, meanN0, numwaves, waveampfactor)

        sps = BSMMiCRMParams(
            cmms_mmicrcm_params,
            Ds,
            CartesianSpace{1,Tuple{Periodic}}(SA[dx]),
            solver_threads
        )

        p = make_smmicrm_problem(sps, u0, T)
        s = solve(p, QNDF();
            dense=false,
            save_everystep=false,
            abstol=tol, reltol=tol,
            callback=make_timer_callback(maxtime),
        )

        sols[i] = if save_sols
            s
        else
            nothing
        end
        retcodes[i] = s.retcode
        fss[i] = s.u[end]
        fTs[i] = s.t[end]
        outcomes[i] = cmms_spatial_sol_to_outcome(s; extthreshold)

        next!(prog)
        flush(stdout)
    end
    finish!(prog)
    flush(stdout)

    (; outcomes, retcodes, fss, fTs, sols)
end
function run_siny_cmms(cmmsp::CMMsParams, args...; kwargs...)
    run_siny_cmms(cmmsp_to_mmicrm(cmmsp), args...; kwargs...)
end

function get_2mm_u0_sines(sN, dx, Nbase, numwaves, faf)
    u0 = fill(0.0, 5, sN)
    N1 = @view u0[1, :]
    N2 = @view u0[2, :]

    N1 .= Nbase
    N2 .= Nbase
    add_1d_many_sines!(N1, numwaves, faf * Nbase / numwaves, dx)
    add_1d_many_sines!(N2, numwaves, faf * Nbase / numwaves, dx)

    clamp!(u0, 0.0, Inf)

    u0
end

################################################################################
# OLD
################################################################################
function make_comp_mms_params(
    K,
    m1, m2,
    l1, l2,
    c1, c2,
    args...;
    d1=1.0, k1=0.0,
    d2=1.0, k2=0.0,
    r=1.0,
)
    D = fill(0.0, 2, 3, 3)
    D[1, 2, 1] = 1.0
    D[2, 3, 1] = 1.0
    BMMiCRMParams(
        fill(1.0, 2), fill(1.0, 3),
        [m1, m2],
        [K, 0.0, 0.0],
        fill(r, 3),
        [l1 k1 0.0; l2 0.0 k2],
        [c1 d1 0.0; c2 0.0 d2], D,
        args...
    )
end

function find_selection_frame(s;
    threshold=10000 * eps(),
    fail_if_extinct=true,
)
    fs = s.u[end]

    loser_i = if fs[1] < threshold && fs[2] < threshold
        if fail_if_extinct
            throw(ArgumentError("This solution has both strains going extinct"))
        else
            return nothing
        end
    elseif fs[1] < fs[2]
        1
    else
        2
    end

    du = fill(0.0, 5)
    ii = length(s.u)
    while ii > 1
        s.prob.f(du, s.u[ii], s.prob.p, s.t[ii])
        if du[loser_i] > threshold
            return ii
        end
        ii -= 1
    end
    return nothing
end

function make_nospace_initial_cond_pds(
    K,
    m1, c1,
    m2, c2;
    T=1e6,
    Ns=range(0.1, 5.0, 100),
    tol=1e-8,
    place=nothing,
)
    if (m1 / c1) <= (m2 / c2)
        println("MM1 is better at SS")
    else
        println("MM2 is better at SS")
    end

    ps = make_comp_mms_params(K, m1, m2, 1.0, 1.0, c1, c2)

    winners = Matrix{Int}(undef, length(Ns), length(Ns))
    N10s = Matrix{Float64}(undef, length(Ns), length(Ns))
    N1ss = Matrix{Union{Missing,Float64}}(undef, length(Ns), length(Ns))
    N1fs = Matrix{Float64}(undef, length(Ns), length(Ns))
    N20s = Matrix{Float64}(undef, length(Ns), length(Ns))
    N2ss = Matrix{Union{Missing,Float64}}(undef, length(Ns), length(Ns))
    N2fs = Matrix{Float64}(undef, length(Ns), length(Ns))
    sTs = Matrix{Union{Missing,Float64}}(undef, length(Ns), length(Ns))
    @tasks for i in 1:length(Ns)
        for j in 1:length(Ns)
            N1 = Ns[i]
            N2 = Ns[j]

            if N1 <= 0.0
                winners[i, j] = 2
                continue
            elseif N2 <= 0.0
                winners[i, j] = 1
                continue
            end


            p = make_mmicrm_problem(ps, [N1, N2, 0.0, 0.0, 0.0], T)
            s = solve(p, QNDF();
                abstol=tol, reltol=tol,
            )

            winners[i, j] = if (s.u[end][1] + s.u[end][2]) < 1e-9
                0
            elseif s.u[end][1] > s.u[end][2]
                1
            else
                2
            end

            N10s[i, j] = N1
            N20s[i, j] = N2
            N1fs[i, j] = s.u[end][1]
            N2fs[i, j] = s.u[end][2]

            ii = find_selection_frame(s;
                threshold=1e-6,
                fail_if_extinct=false,
            )
            if isnothing(ii)
                N1ss[i, j] = missing
                N2ss[i, j] = missing
                sTs[i, j] = missing
            else
                # @show length(s.u)-ii
                us = s.u[ii]
                N1ss[i, j] = us[1]
                N2ss[i, j] = us[2]
                sTs[i, j] = s.t[ii]
            end
        end
    end

    chi1s = [(m1 / c1) * ((1.0 / (c1 * N)) + 1) for N in N1ss]
    chi2s = [(m2 / c2) * ((1.0 / (c2 * N)) + 1) for N in N2ss]
    bchi1s = [(m1 / c1) * ((1.0 / (c1 * N)) + 1) for N in N10s]
    bchi2s = [(m2 / c2) * ((1.0 / (c2 * N)) + 1) for N in N20s]

    if isnothing(place)
        place = Figure()
    end
    ax = Axis(place[1, 1];
        xlabel=L"N_1(0)",
        ylabel=L"N_2(0)",
    )

    hm = heatmap!(ax, Ns, Ns, winners;
        colormap=Categorical(:viridis),
    )
    Colorbar(place[1, 2], hm)

    mn = minimum(Ns)
    mx = maximum(Ns)
    lines!(ax, [mn, mx], [mn, mx]; color=:black, linewidth=2)

    # Draw N0 prediction line
    xx = N2_0_crit.(Ns, m1, c1, m2, c2, 1.0, 1.0)
    lines!(ax, Ns, xx)

    # Draw actual Ns prediction line
    yy1 = []
    yy2 = []
    for i in 1:length(Ns)
        for j in 1:length(Ns)
            chi1 = chi1s[i, j]
            chi2 = chi2s[i, j]
            if !ismissing(chi1) && !ismissing(chi2) && chi2 < chi1
                push!(yy1, Ns[i])
                push!(yy2, Ns[j])
                break
            end
        end
    end
    if !isempty(yy1)
        lines!(ax, yy1, yy2)
    end

    #

    xlims!(ax, extrema(Ns)...)
    ylims!(ax, extrema(Ns)...)

    nothing
end
