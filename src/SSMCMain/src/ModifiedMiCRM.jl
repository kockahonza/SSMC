"""Modified MiCRM model"""
module ModifiedMiCRM

using Reexport
@reexport using ..SSMCMain

################################################################################
# Internals
################################################################################
struct MMiCRMParams{Ns,Nr,F} # number of strains and resource types
    # these are usually all 1 from dimensional reduction
    g::SVector{Ns,F}
    w::SVector{Nr,F}

    # strain props
    m::SVector{Ns,F}

    # resource props
    K::SVector{Nr,F}
    r::SVector{Nr,F}

    # complex, matrix params
    l::SMatrix{Ns,Nr,F}
    c::SMatrix{Ns,Nr,F}
    D::SArray{Tuple{Ns,Nr,Nr},F} # D[1,a,b] corresponds to b -> a
end
get_Ns(_::MMiCRMParams{Ns,Nr}) where {Ns,Nr} = (Ns, Nr)
function mmicrmfunc!(du, u, p::MMiCRMParams{Ns,Nr}, _=0) where {Ns,Nr}
    N = @view u[1:Ns]
    R = @view u[Ns+1:Ns+Nr]
    dN = @view du[1:Ns]
    dR = @view du[Ns+1:Ns+Nr]

    for i in 1:Ns
        sumterm = 0.0
        for a in 1:Nr
            sumterm += p.w[a] * (1.0 - p.l[i, a]) * p.c[i, a] * R[a]
        end
        dN[i] = p.g[i] * N[i] * (sumterm - p.m[i])
    end

    for a in 1:Nr
        sumterm1 = 0.0
        for i in 1:Ns
            sumterm1 += N[i] * p.c[i, a] * R[a]
        end
        sumterm2 = 0.0
        for i in 1:Ns
            for b in 1:Nr
                sumterm2 += p.D[i, a, b] * (p.w[b] / p.w[a]) * p.l[i, b] * N[i] * p.c[i, b] * R[b]
            end
        end

        dR[a] = p.K[a] - p.r[a] * R[a] - sumterm1 + sumterm2
    end
    du
end
export MMiCRMParams, get_Ns, mmicrmfunc!

#For testing
function trivmmicrmparams(Ns, Nr;
    m=1.0, l=0.0, K=1.0, r=0.0, c=1.0
)
    MMiCRMParams(
        (@SVector fill(1.0, Ns)),
        (@SVector fill(1.0, Nr)),
        (@SVector fill(m, Ns)),
        (@SVector fill(K, Nr)),
        (@SVector fill(r, Nr)),
        (@SMatrix fill(l, Ns, Nr)),
        (@SMatrix fill(c, Ns, Nr)),
        (@SArray fill(1 / Nr, Ns, Nr, Nr))
    )
end
export trivmmicrmparams
#
# ################################################################################
# # Smart functions
# ################################################################################
"""Smart function for making the params"""
function make_mmicrmparams_smart(Ns, Nr;
    g=nothing, w=nothing,
    m=nothing, l=nothing, K=nothing, r=nothing,
    c=:uniform, D=:euniform
)
    # Setup the MMiCRMParams
    if isa(c, Number) || isa(c, AbstractArray)
        c = smart_sval(c, 0.0, Ns, Nr)
    else
        cname, cargs = split_name_args(c)

        if cname == :uniform
            c = make_c_uniform(Ns, Nr, cargs...)
        elseif cname == :oto
            c = make_c_oto(Ns, Nr, cargs...)
        else
            throw(ArgumentError(@sprintf "cannot correctly parse cname %s" string(cname)))
        end
    end

    if isa(D, Number) || isa(D, AbstractArray)
        D = smart_sval(D, nothing, Ns, Nr, Nr)
    else
        Dname, Dargs = split_name_args(D)

        if Dname == :uniform
            D = make_D_uniform(Ns, Nr, Dargs...)
        elseif Dname == :euniform
            D = make_D_euniform(Ns, Nr, Dargs...)
        else
            throw(ArgumentError(@sprintf "cannot correctly parse Dname %s" string(Dname)))
        end
    end

    MMiCRMParams(
        smart_sval(g, 1.0, Ns),
        smart_sval(w, 1.0, Nr),
        smart_sval(m, 1.0, Ns),
        smart_sval(K, 1.0, Nr),
        smart_sval(r, 1.0, Nr),
        smart_sval(l, 0.0, Ns, Nr),
        c, D
    )
end
make_c_uniform(Ns, Nr, val=1.0) = @SMatrix fill(val, Ns, Nr)
function make_c_oto(Ns, Nr, val=1.0) # strain i eats resource i if it exists
    mat = zeros(Ns, Nr)
    for i in 1:min(Ns, Nr)
        mat[i, i] = val
    end
    SMatrix{Ns,Nr}(mat)
end
make_D_uniform(Ns, Nr) = @SArray fill(1 / Nr, Ns, Nr, Nr)
function make_D_euniform(Ns, Nr)
    if Nr == 1
        @warn "using euniform D with only 1 resource, this violates certain assumptions"
    end
    mat = fill(1 / (Nr - 1), Ns, Nr, Nr)
    for i in 1:Ns
        for a in 1:Nr
            mat[i, a, a] = 0.0
        end
    end
    SArray{Tuple{Ns,Nr,Nr}}(mat)
end

"""Smart function for making an initial state"""
function make_mmicrmu0_smart(params::MMiCRMParams{Ns,Nr};
    u0=:steadyR, u0rand=nothing
) where {Ns,Nr}
    if isa(u0, Number) || isa(u0, AbstractArray)
        u0 = smart_val(u0, make_u0_steadyR(params), Ns + Nr)
    else
        u0name, u0args = split_name_args(u0)

        if u0name == :uniE
            u0 = make_u0_uniE(params)
        elseif u0name == :steadyR
            u0 = make_u0_steadyR(params)
        elseif u0name == :onlyN
            u0 = make_u0_onlyN(params)
        else
            throw(ArgumentError(@sprintf "cannot correctly parse u0name %s" string(u0name)))
        end
    end

    if !isnothing(u0rand)
        for i in eachindex(u0)
            u0[i] *= 1 + u0rand * (2 * rand() - 1)
        end
    end

    u0
end
make_u0_uniE(p::MMiCRMParams{Ns,Nr}) where {Ns,Nr} = Vector(vcat(p.g, 1.0 ./ p.w))
make_u0_steadyR(p::MMiCRMParams{Ns,Nr}) where {Ns,Nr} = Vector(vcat(p.g, p.K ./ p.r))
make_u0_onlyN(p::MMiCRMParams{Ns,Nr}) where {Ns,Nr} = vcat(p.g, fill(0.0, Nr))

"""Make the problem directly"""
function make_mmicrm_smart(Ns, Nr, T=1.0;
    u0=:steadyR, u0rand=nothing, kwargs...
)
    params = make_mmicrmparams_smart(Ns, Nr; kwargs...)
    u0 = make_mmicrmu0_smart(params; u0, u0rand)

    # make problem
    ODEProblem(mmicrmfunc!, u0, (0.0, T), params)
end
export make_mmicrm_smart, make_mmicrmparams_smart, make_mmicrmu0_smart

function plot_mmicrm_sol(sol; singleax=false, plote=false)
    params = sol.prob.p
    if !isa(params, MMiCRMParams)
        throw(ArgumentError("plot_mmicrm_sol can only plot solutions of MMiCRM problems"))
    end
    Ns, Nr = get_Ns(params)

    fig = Figure()
    if singleax
        strainax = resax = Axis(fig[1, 1])
        if plote
            eax = strainax
        end
    else
        strainax = Axis(fig[1, 1])
        resax = Axis(fig[2, 1])
        if plote
            eax = Axis(fig[3, 1])
        end
    end

    # plot data
    for i in 1:Ns
        lines!(strainax, sol.t, sol[i, :]; label=@sprintf "str %d" i)
    end
    for a in 1:Nr
        lines!(resax, sol.t, sol[Ns+a, :]; label=@sprintf "res %d" a)
    end
    if plote
        # TODO: uncomment
        # lines!(eax, sol.t, calc_E.(sol.u, Ref(params)); label=L"\epsilon")
    end

    if singleax
        axislegend(strainax)
    else
        axislegend(strainax)
        axislegend(resax)
        if plote
            axislegend(eax)
        end
    end
    fig
end
export plot_mmicrm_sol
#
function make_solve_plot_return(args...; kwargs...)
    p = make_mmicrm_smart(args...; kwargs...)
    s = solve(p)
    display(plot_mmicrm_sol(s))
    p, s
end
export make_solve_plot_return

################################################################################
# Linear stability analysis
################################################################################
function do_linstab_for_ks(ks, p::MMiCRMParams{Ns,Nr,F}, Ds, ss; kwargs...) where {Ns,Nr,F}
    lambda_func = linstab_make_lambda_func(p, ss, Ds; kwargs...)
    lambdas = Matrix{Complex{F}}(undef, length(ks), length(ss))
    for (i, k) in enumerate(ks)
        lambdas[i, :] .= lambda_func(k)
    end
    lambdas
end
function do_linstab_for_ks(ks, p::ODEProblem, Ds, ss=nothing; kwargs...)
    if isnothing(ss)
        ssprob = SteadyStateProblem(p)
        sssol = solve(ssprob, DynamicSS())
        if sssol.retcode != ReturnCode.Success
            @error "the steady state solver did not succeed"
        end
        ss = sssol.u
    end
    do_linstab_for_ks(ks, p.p, Ds, ss; kwargs...)
end
export do_linstab_for_ks

function linstab_make_lambda_func(p::MMiCRMParams{Ns,Nr}, ss, Ds=nothing; kwargs...) where {Ns,Nr}
    if !isnothing(Ds)
        let M1 = make_M1(p, ss)
            function (k)
                eigvals(M1 + Diagonal(-(k^2) .* Ds); kwargs...)
            end
        end
    else
        let M1 = make_M1(p, ss)
            function (k, Ds)
                eigvals(M1 + Diagonal(-(k^2) .* Ds); kwargs...)
            end
        end
    end
end
export linstab_make_lambda_func

function linstab_benchmark(p::MMiCRMParams, ss; kwargs...)
end
export linstab_benchmark

function make_M1!(M1, p::MMiCRMParams{Ns,Nr}, ss) where {Ns,Nr}
    Nss = @view ss[1:Ns]
    Rss = @view ss[Ns+1:Ns+Nr]

    for i in 1:Ns
        # 0 everywhere
        for j in 1:Ns
            M1[i, j] = 0.0
        end
        # add things to the diagonal - this is T
        for a in 1:Nr
            M1[i, i] += p.g[i] * (1 - p.l[i, a]) * p.w[a] * p.c[i, a] * Rss[a]
        end
        M1[i, i] -= p.g[i] * p.m[i]
    end
    for i in 1:Ns
        for a in 1:Nr
            # this is U
            M1[i, Ns+a] = p.g[i] * (1 - p.l[i, a]) * p.w[a] * p.c[i, a] * Nss[i]
            # this is V
            M1[Ns+a, i] = 0.0
            for b in 1:Nr
                M1[Ns+a, i] += p.D[i, a, b] * p.l[i, b] * (p.w[b] / p.w[a]) * p.c[i, b] * Rss[b]
            end
            M1[Ns+a, i] -= p.c[i, a] * Rss[a]
        end
    end
    for a in 1:Nr
        for b in 1:Nr
            # makes sure everything is initialized
            M1[Ns+a, Ns+b] = 0.0
            # this does the complex part of W
            for i in 1:Ns
                M1[Ns+a, Ns+b] += p.D[i, a, b] * p.l[i, b] * (p.w[b] / p.w[a]) * p.c[i, b] * Nss[i]
            end
        end
        # and add the diagonal bit of W
        for i in 1:Ns
            M1[Ns+a, Ns+a] -= p.c[i, a] * Nss[i]
        end
        M1[Ns+a, Ns+a] -= p.r[a]
    end
end
function make_M1(p::MMiCRMParams{Ns,Nr,F}, args...) where {Ns,Nr,F}
    M1 = Matrix{F}(undef, Ns + Nr, Ns + Nr)
    make_M1!(M1, p, args...)
    M1
end
export make_M1!, make_M1

end
