################################################################################
# Base Array MMiCRMParams
################################################################################
struct BMMiCRMParams{P<:Union{Nothing,Int},F} <: AbstractMMiCRMParams{F}
    # these are usually all 1 from dimensional reduction
    g::Vector{F}
    w::Vector{F}

    # strain props
    m::Vector{F}

    # resource props
    K::Vector{F}
    r::Vector{F}

    # complex, matrix params
    l::Matrix{F}
    c::Matrix{F}
    D::Array{F,3} # D[i,a,b] corresponds to b -> a

    usenthreads::P

    function BMMiCRMParams(g, w, m, K, r, l, c, D, usenthreads=nothing)
        if (ndims(g) != 1) ||
           (ndims(w) != 1) ||
           (ndims(m) != 1) ||
           (ndims(K) != 1) ||
           (ndims(r) != 1) ||
           (ndims(l) != 2) ||
           (ndims(c) != 2) ||
           (ndims(D) != 3)
            throw(ArgumentError("arguments passed to BMMiCRMParams constructor do not have the correct dimensionality"))
        end

        Ns = length(g)
        Nr = length(w)

        if (size(m)[1] != Ns) ||
           (size(K)[1] != Nr) ||
           (size(r)[1] != Nr) ||
           (size(l) != (Ns, Nr)) ||
           (size(c) != (Ns, Nr)) ||
           (size(D) != (Ns, Nr, Nr))
            throw(ArgumentError("arguments passed to BMMiCRMParams constructor do not have matching sizes"))
        end

        new{typeof(usenthreads),eltype(g)}(g, w, m, K, r, l, c, D, usenthreads)
    end
end
get_Ns(p::BMMiCRMParams) = (length(p.g), length(p.w))
function mmicrmfunc!(du, u, p::BMMiCRMParams{Nothing}, t=0)
    Ns, Nr = get_Ns(p)

    @inbounds for i in 1:Ns
        sumterm = 0.0
        for a in 1:Nr
            sumterm += p.w[a] * (1.0 - p.l[i, a]) * p.c[i, a] * u[Ns+a]
        end
        du[i] = p.g[i] * u[i] * (sumterm - p.m[i])
    end

    @inbounds for a in 1:Nr
        sumterm1 = 0.0
        for i in 1:Ns
            sumterm1 += u[i] * p.c[i, a] * u[Ns+a]
        end
        sumterm2 = 0.0
        for i in 1:Ns
            for b in 1:Nr
                sumterm2 += p.D[i, a, b] * (p.w[b] / p.w[a]) * p.l[i, b] * u[i] * p.c[i, b] * u[Ns+b]
            end
        end

        du[Ns+a] = p.K[a] - p.r[a] * u[Ns+a] - sumterm1 + sumterm2
    end
    du
end
function mmicrmfunc!(du, u, p::BMMiCRMParams{Int}, t=0)
    Ns, Nr = get_Ns(p)

    @tasks for i in 1:Ns
        @set ntasks = p.usenthreads
        sumterm = 0.0
        @inbounds for a in 1:Nr
            sumterm += p.w[a] * (1.0 - p.l[i, a]) * p.c[i, a] * u[Ns+a]
        end
        @inbounds du[i] = p.g[i] * u[i] * (sumterm - p.m[i])
    end

    @tasks for a in 1:Nr
        @set ntasks = p.usenthreads
        sumterm1 = 0.0
        @inbounds for i in 1:Ns
            sumterm1 += u[i] * p.c[i, a] * u[Ns+a]
        end
        sumterm2 = 0.0
        @inbounds for i in 1:Ns
            for b in 1:Nr
                sumterm2 += p.D[i, a, b] * (p.w[b] / p.w[a]) * p.l[i, b] * u[i] * p.c[i, b] * u[Ns+b]
            end
        end

        @inbounds du[Ns+a] = p.K[a] - p.r[a] * u[Ns+a] - sumterm1 + sumterm2
    end
    du
end
function copy(p::BMMiCRMParams)
    BMMiCRMParams(
        copy(p.g), copy(p.w), copy(p.m), copy(p.K), copy(p.r),
        copy(p.l), copy(p.c), copy(p.D), p.usenthreads
    )
end
export BMMiCRMParams

struct BSMMiCRMParams{S<:Union{Nothing,AbstractSpace},P<:Union{Nothing,Int},P2,F} <: AbstractSMMiCRMParams{S,F}
    mmicrm_params::BMMiCRMParams{P2,F}
    Ds::Vector{F}
    space::S

    usenthreads::P

    function BSMMiCRMParams(
        mmicrm_params::BMMiCRMParams{P2,F},
        Ds,
        space=nothing,
        usenthreads=nothing,
    ) where {P2,F}
        if length(Ds) == sum(get_Ns(mmicrm_params))
            new{typeof(space),typeof(usenthreads),P2,F}(mmicrm_params, Ds, space, usenthreads)
        else
            throw(ArgumentError("passed mmicrm_params and diff are not compatible"))
        end
    end
end
get_Ns(sp::BSMMiCRMParams) = get_Ns(sp.mmicrm_params)
mmicrmfunc!(du, u, sp::BSMMiCRMParams, t=0) = mmicrmfunc!(du, u, sp.mmicrm_params, t)
function getproperty(sp::BSMMiCRMParams, sym::Symbol, args...)
    if sym in (:g, :w, :m, :K, :r, :l, :c, :D)
        getproperty(sp.mmicrm_params, sym, args...)
    else
        getfield(sp, sym, args...)
    end
end
get_Ds(sp::BSMMiCRMParams) = sp.Ds
get_space(sp::BSMMiCRMParams) = sp.space
function smmicrmfunc!(du, u, p::BSMMiCRMParams, t=0)
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
function copy(sp::BSMMiCRMParams)
    BSMMiCRMParams(copy(sp.mmicrm_params), copy(sp.Ds), copy(sp.space), sp.usenthreads)
end
export BSMMiCRMParams

################################################################################
# Helper functions
################################################################################
"""Smart function for making the params"""
function make_bmmicrmparams(Ns, Nr;
    g=nothing, w=nothing,
    m=nothing, l=nothing, K=nothing, r=nothing,
    c=:uniform, D=:euniform,
    usenthreads=nothing
)
    # Setup the MMiCRMParams
    if isa(c, Number) || isa(c, AbstractArray)
        c = smart_val(c, 0.0, Ns, Nr)
    else
        cname, cargs = split_name_args(c)

        val = isempty(cargs) ? 1.0 : cargs[1]
        if cname == :uniform
            c = fill(val, Ns, Nr)
        elseif cname == :oto
            c = zeros(Ns, Nr)
            for i in 1:min(Ns, Nr)
                c[i, i] = val
            end
        else
            throw(ArgumentError(@sprintf "cannot correctly parse cname %s" string(cname)))
        end
    end

    if isa(D, Number) || isa(D, AbstractArray)
        D = smart_val(D, nothing, Ns, Nr, Nr)
    else
        Dname, Dargs = split_name_args(D)

        if Dname == :uniform
            D = fill(1 / Nr, Ns, Nr, Nr)
        elseif Dname == :euniform
            if Nr == 1
                @warn "using euniform D with only 1 resource, this violates certain assumptions"
            end
            D = fill(1 / (Nr - 1), Ns, Nr, Nr)
            for i in 1:Ns
                for a in 1:Nr
                    D[i, a, a] = 0.0
                end
            end
        else
            throw(ArgumentError(@sprintf "cannot correctly parse Dname %s" string(Dname)))
        end
    end

    BMMiCRMParams(
        smart_val(g, 1.0, Ns),
        smart_val(w, 1.0, Nr),
        smart_val(m, 1.0, Ns),
        smart_val(K, 1.0, Nr),
        smart_val(r, 1.0, Nr),
        smart_val(l, 0.0, Ns, Nr),
        c, D,
        usenthreads
    )
end
export make_bmmicrmparams

function change_bmmicrmparams(p::BMMiCRMParams;
    g=copy(p.g), w=copy(p.w),
    m=copy(p.m), K=copy(p.K), r=copy(p.r),
    l=copy(p.l), c=copy(p.c), D=copy(p.D),
    usenthreads=p.usenthreads
)
    BMMiCRMParams(g, w, m, K, r, l, c, D, usenthreads)
end
export change_bmmicrmparams

function change_bsmmicrmparams(sp::BSMMiCRMParams;
    mmicrm_params=nothing,
    Ds=sp.Ds,
    space=sp.space,
    usenthreads=sp.usenthreads,
    nospace_usenthreads=false,
    kwargs...
)
    if isnothing(mmicrm_params)
        mmicrm_params = sp.mmicrm_params
    end
    if !isempty(kwargs) || (nospace_usenthreads != false)
        if nospace_usenthreads == false
            mmicrm_params = change_bmmicrmparams(mmicrm_params; kwargs...)
        else
            mmicrm_params = change_bmmicrmparams(mmicrm_params; kwargs..., usenthreads=nospace_usenthreads)
        end
    end
    BSMMiCRMParams(mmicrm_params, Ds, space, usenthreads)
end
export change_bsmmicrmparams
