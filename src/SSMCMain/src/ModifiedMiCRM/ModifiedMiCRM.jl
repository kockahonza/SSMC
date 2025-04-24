"""Modified MiCRM model"""
module ModifiedMiCRM

using Reexport
@reexport using ..SSMCMain

using Polynomials

################################################################################
# Internals
################################################################################
struct MMiCRMParams{Ns,Nr,F,A,B} # number of strains and resource types
    # these are usually all 1 from dimensional reduction
    g::SVector{Ns,F}
    w::SVector{Nr,F}

    # strain props
    m::SVector{Ns,F}

    # resource props
    K::SVector{Nr,F}
    r::SVector{Nr,F}

    # complex, matrix params
    l::SMatrix{Ns,Nr,F,A}
    c::SMatrix{Ns,Nr,F,A}
    D::SArray{Tuple{Ns,Nr,Nr},F,3,B} # D[1,a,b] corresponds to b -> a
end
get_Ns(_::MMiCRMParams{Ns,Nr}) where {Ns,Nr} = (Ns, Nr)
function mmicrmfunc!(du, u, p::MMiCRMParams{Ns,Nr}, _=0) where {Ns,Nr}
    N = @view u[1:Ns]
    R = @view u[Ns+1:Ns+Nr]
    dN = @view du[1:Ns]
    dR = @view du[Ns+1:Ns+Nr]

    @inbounds for i in 1:Ns
        sumterm = 0.0
        for a in 1:Nr
            sumterm += p.w[a] * (1.0 - p.l[i, a]) * p.c[i, a] * R[a]
        end
        dN[i] = p.g[i] * N[i] * (sumterm - p.m[i])
    end

    @inbounds for a in 1:Nr
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

function check_mmicrmparams(p::MMiCRMParams{Ns,Nr}) where {Ns,Nr}
    # check D
    for i in 1:Ns
        for a in 1:Nr
            total_out = 0.0
            for b in 1:Nr
                total_out += p.D[i, b, a]
            end
            if total_out > 1.0
                @error (@sprintf "strain %d leaks more than it energetically can through consuming %d" i a)
            end
            if (total_out < 1.0) && (p.l[i, a] != 0)
                @info (@sprintf "strain %d leaks less than it energetically can through consuming %d" i a)
            end
        end
    end
end
export check_mmicrmparams

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

################################################################################
# Smart functions
################################################################################
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

################################################################################
# Imports
################################################################################
# Linear stability analysis
include("linstab.jl")
#
# Plotting
include("mmicrm_plotting.jl")

# Physicsy bits
include("mmicrm_physics.jl")

end
