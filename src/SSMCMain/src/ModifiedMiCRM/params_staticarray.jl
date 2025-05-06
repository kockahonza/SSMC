################################################################################
# Static Static Array MMiCRMParams
################################################################################
struct SAMMiCRMParams{Ns,Nr,F,A,B} <: AbstractMMiCRMParams{F} # number of strains and resource types
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
get_Ns(_::SAMMiCRMParams{Ns,Nr}) where {Ns,Nr} = (Ns, Nr)
function mmicrmfunc!(du, u, p::SAMMiCRMParams{Ns,Nr}, t=0) where {Ns,Nr}
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
export SAMMiCRMParams

struct SASMMiCRMParams{Ns,Nr,S<:Union{Nothing,AbstractSpace},F,N,P<:Union{Nothing,Int},A,B} <: AbstractSMMiCRMParams{S,F}
    mmicrm_params::SAMMiCRMParams{Ns,Nr,F,A,B}
    Ds::SVector{N,F}
    space::S

    usenthreads::P

    function SASMMiCRMParams(
        mmicrm_params::SAMMiCRMParams{Ns,Nr,F,A,B},
        diff::SVector{N,F},
        space::S=nothing,
        usenthreads=nothing
    ) where {Ns,Nr,S,F,N,A,B}
        if N == (Ns + Nr)
            new{Ns,Nr,S,F,N,typeof(usenthreads),A,B}(mmicrm_params, diff, space, usenthreads)
        else
            throw(ArgumentError("passed mmicrm_params and diff are not compatible"))
        end
    end
end
get_Ns(_::SASMMiCRMParams{Ns,Nr}) where {Ns,Nr} = (Ns, Nr)
mmicrmfunc!(du, u, sp::SASMMiCRMParams, t=0) = mmicrmfunc!(du, u, sp.mmicrm_params, t)
function getproperty(sp::SASMMiCRMParams, sym::Symbol, args...)
    if sym in (:g, :w, :m, :K, :r, :l, :c, :D)
        getproperty(sp.mmicrm_params, sym, args...)
    else
        getfield(sp, sym, args...)
    end
end
get_Ds(sp::SASMMiCRMParams) = sp.Ds
get_space(sp::SASMMiCRMParams) = sp.space
function smmicrmfunc!(du, u, p::SASMMiCRMParams, t=0)
    if isnothing(p.usenthreads)
        @inbounds for r in CartesianIndices(axes(u)[2:end])
            mmicrmfunc!((@view du[:, r]), (@view u[:, r]), p.mmicrm_params, t)
        end
    else
        rchunks = chunks(CartesianIndices(axes(u)[2:end]); n=p.usenthreads)
        @sync for rs in rchunks
            @spawn begin
                @inbounds for r in rs
                    mmicrmfunc!((@view du[:, r]), (@view u[:, r]), p.mmicrm_params, t)
                end
            end
        end
    end
    add_diffusion!(du, u, p.Ds, p.space, p.usenthreads)
    du
end
export SASMMiCRMParams

################################################################################
# Helper functions
################################################################################
"""Smart function for making the params"""
function make_sammicrmparams(Ns, Nr;
    g=nothing, w=nothing,
    m=nothing, l=nothing, K=nothing, r=nothing,
    c=:uniform, D=:euniform
)
    # Setup the SAMMiCRMParams
    if isa(c, Number) || isa(c, AbstractArray)
        c = smart_sval(c, 0.0, Ns, Nr)
    else
        cname, cargs = split_name_args(c)

        val = isempty(cargs) ? 1.0 : cargs[1]
        if cname == :uniform
            c = @SMatrix fill(val, Ns, Nr)
        elseif cname == :oto
            mat = zeros(Ns, Nr)
            for i in 1:min(Ns, Nr)
                mat[i, i] = val
            end
            c = SMatrix{Ns,Nr}(mat)
        else
            throw(ArgumentError(@sprintf "cannot correctly parse cname %s" string(cname)))
        end
    end

    if isa(D, Number) || isa(D, AbstractArray)
        D = smart_sval(D, nothing, Ns, Nr, Nr)
    else
        Dname, Dargs = split_name_args(D)

        if Dname == :uniform
            D = @SArray fill(1 / Nr, Ns, Nr, Nr)
        elseif Dname == :euniform
            if Nr == 1
                @warn "using euniform D with only 1 resource, this violates certain assumptions"
            end
            mat = fill(1 / (Nr - 1), Ns, Nr, Nr)
            for i in 1:Ns
                for a in 1:Nr
                    mat[i, a, a] = 0.0
                end
            end
            D = SArray{Tuple{Ns,Nr,Nr}}(mat)
        else
            throw(ArgumentError(@sprintf "cannot correctly parse Dname %s" string(Dname)))
        end
    end

    SAMMiCRMParams(
        smart_sval(g, 1.0, Ns),
        smart_sval(w, 1.0, Nr),
        smart_sval(m, 1.0, Ns),
        smart_sval(K, 1.0, Nr),
        smart_sval(r, 1.0, Nr),
        smart_sval(l, 0.0, Ns, Nr),
        c, D
    )
end
export make_sammicrmparams

function change_sasmmicrm_params(sp::SASMMiCRMParams;
    mmicrm_params=sp.mmicrm_params,
    Ds=sp.Ds,
    space=sp.space,
    usenthreads=sp.usenthreads
)
    SASMMiCRMParams(mmicrm_params, Ds, space, usenthreads)
end
export change_sasmmicrm_params
