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
export BMMiCRMParams

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
    # Setup the SAMMiCRMParams
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

function change_bmmicrmparams(ps::BMMiCRMParams;
    g=copy(ps.g), w=copy(ps.w),
    m=copy(ps.m), K=copy(ps.K), r=copy(ps.r),
    l=copy(ps.l), c=copy(ps.c), D=copy(ps.D),
    usenthreads=ps.usenthreads
)
    BMMiCRMParams(g, w, m, K, r, l, c, D, usenthreads)
end
export change_bmmicrmparams
