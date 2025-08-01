################################################################################
# Marsland samplers
################################################################################
Base.@kwdef struct MarslandMatrixSampler
    # Define the strain and resource families and types
    T::Int                       # how many resource types there are
    M_per_type::Int = 1          # resources per type
    Mwaste::Int = 0              # number of waste resources (a special type)
    F::Int                       # number of strain specialist families (each have their preferred resource type to eat) 0 <= F <= T
    S_per_family::Int = 1        # number of strains per family
    Sgen::Int = 0                # number of generalist strains (they have no preffered resource)
    # These are computed automatically - total number of resources and strains
    M = T * M_per_type + Mwaste
    S = F * S_per_family + Sgen
    # The rest are exactly as taken from the Marsland paper
    # This sets the cs
    muc::Float64 = 1.0
    c0::Float64 = 0.0
    c1::Float64 = 1.0
    q::Float64 = 1.0
    # This sets the Ds
    s::Float64 = 1.0
    fs::Float64 = 0.45
    fw::Float64 = 0.45
end
function (mms::MarslandMatrixSampler)()
    Sfamilies = fill(-1, mms.S) # generalists by default
    for fi in 1:mms.F
        for i in 1:mms.S_per_family
            Sfamilies[(fi-1)*mms.S_per_family+i] = fi
        end
    end

    Mtypes = fill(-1, mms.M)
    for ti in 1:mms.T
        for i in 1:mms.M_per_type
            Mtypes[(ti-1)*mms.M_per_type+i] = ti
        end
    end

    # Start generating
    ####################
    # make c based on the paper
    c = fill(0.0, mms.S, mms.M)
    for i in 1:mms.S
        for a in 1:mms.M
            a_is_waste = Mtypes[a] == -1
            if Sfamilies[i] == -1 # this species is a generalist
                p = mms.muc / (mms.M * mms.c1)
            else
                if !a_is_waste && (Sfamilies[i] == Mtypes[a])
                    p = (mms.muc / (mms.M * mms.c1)) * (1 + mms.q * (mms.M - mms.M_per_type) / mms.M_per_type)
                else
                    p = (mms.muc / (mms.M * mms.c1)) * (1 - mms.q)
                end
            end
            X = rand(Bernoulli(p))
            c[i, a] = mms.c0 / mms.M + mms.c1 * X
        end
    end

    # make the original Dab (not Diab!)
    D_noi = fill(0.0, mms.M, mms.M)
    for b in 1:mms.M
        ds = Float64[]
        for g in 1:mms.M
            fb = Mtypes[b]
            fg = Mtypes[g]

            d = if (fb != -1) && (fg == -1)
                mms.fw / (mms.s * mms.Mwaste)
            elseif (fb != -1) && (fb == fg)
                mms.fs / (mms.s * mms.M_per_type)
            elseif (fb != -1) && (fg != -1) && (fb != fg)
                (1 - mms.fs - mms.fw) / (mms.s * (mms.M - mms.M_per_type - mms.Mwaste))
            elseif (fb == -1) && (fg == -1)
                (mms.fw + mms.fs) / (mms.s * mms.Mwaste)
            elseif (fb == -1) && (fg != -1)
                (1 - mms.fw - mms.fs) / (mms.s * (mms.M - mms.Mwaste))
            else
                throw(ErrorException("Unexpected case in type-family association"))
            end

            push!(ds, d)
        end

        D_noi[:, b] .= rand(Dirichlet(ds))
    end

    c, D_noi
end
export MarslandMatrixSampler

struct TrueMarslandSampler
    Ns::Int # corresponds to S
    Nr::Int # corresponds to M
    mms::MarslandMatrixSampler # used for generating the c and D matrices just as Marsland does

    # Marsland sets these all to be a single value, but lets keep it general for now
    m::UnivariateDistribution
    r::UnivariateDistribution
    l::UnivariateDistribution

    # I think this is how they do the Ks - it's always the first resource that is supplied
    # and the K value is set via setting the "supply resource level" R0 
    food_R::Float64

    # diffusion constant distrubutions
    Ds::UnivariateDistribution
    Dr::UnivariateDistribution

    usenthreads::Union{Nothing,Int}

    function TrueMarslandSampler(;
        m=1.0, r=1.0, l=0.0, Ds=1e-10, Dr=1.0,
        food_R=0.0,
        usenthreads=nothing,
        kwargs...,
    )
        mms = MarslandMatrixSampler(; kwargs...)
        Ns = mms.S
        Nr = mms.M

        if isa(m, Number)
            m = Dirac(m)
        elseif isa(m, Tuple) && length(m) == 2
            m = Normal(m[1], m[2])
        end
        if isa(r, Number)
            r = Dirac(r)
        elseif isa(r, Tuple) && length(r) == 2
            r = Normal(r[1], r[2])
        end
        if isa(l, Number)
            l = Dirac(l)
        elseif isa(l, Tuple) && length(l) == 2
            l = Normal(l[1], l[2])
        end
        if isa(Ds, Number)
            Ds = Dirac(Ds)
        elseif isa(Ds, Tuple) && length(Ds) == 2
            Ds = Normal(Ds[1], Ds[2])
        end
        if isa(Dr, Number)
            Dr = Dirac(Dr)
        elseif isa(Dr, Tuple) && length(Dr) == 2
            Dr = Normal(Dr[1], Dr[2])
        end

        new(Ns, Nr, mms, m, r, l, food_R, Ds, Dr, usenthreads)
    end
end
function (ms::TrueMarslandSampler)()
    g = fill(1.0, ms.Ns)
    w = fill(1.0, ms.Nr)

    m = clamp.(rand(ms.m, ms.Ns), 0.0, Inf)
    r = clamp.(rand(ms.r, ms.Nr), 0.0, Inf)

    c, Dnoi = ms.mms()
    lnoi = [clamp(rand(ms.l), 0.0, 1.0) for _ in 1:ms.Nr]
    l = fill(0.0, (ms.Ns, ms.Nr))
    D = fill(0.0, (ms.Ns, ms.Nr, ms.Nr))
    for i in 1:ms.Ns
        for a in 1:ms.Nr
            c_ = c[i, a]
            if !iszero(c_)
                l[i, a] = lnoi[a]
                D[i, :, a] = Dnoi[:, a]
            end
        end
    end

    K = fill(0.0, ms.Nr)
    K[1] = ms.food_R / r[1]

    Ds = clamp.(rand(ms.Ds, ms.Ns), 0.0, Inf)
    Dr = clamp.(rand(ms.Dr, ms.Nr), 0.0, Inf)
    Ds = vcat(Ds, Dr)

    mmicrm_params = BMMiCRMParams(g, w, m, K, r, l, c, D, ms.usenthreads)
    BSMMiCRMParams(mmicrm_params, Ds)
end
export TrueMarslandSampler

struct AdaptedMarsland1
    Ns::Int # corresponds to S
    Nr::Int # corresponds to M

    T::Int
    M_per_type::Int
    Mwaste::Int
    F::Int
    S_per_family::Int
    Sgen::Int

    # maps individual strains/resources to their families/types
    Sfamilies::Vector{Int}
    Mtypes::Vector{Int}

    # This sets the cs
    muc::Float64
    c0::Float64
    c1::Float64
    q::Float64
    # This sets the Ds
    s::Float64
    fs::Float64
    fw::Float64

    # Marsland sets these all to be a single value, but lets keep it general for now
    m::UnivariateDistribution
    r::UnivariateDistribution
    l::UnivariateDistribution

    # I think this is how they do the Ks - it's always the first resource that is supplied
    # and the K value is set via setting the "supply resource level" R0 
    food_R::Float64

    # diffusion constant distrubutions
    Ds::UnivariateDistribution
    Dr::UnivariateDistribution

    usenthreads::Union{Nothing,Int}

    function AdaptedMarsland1(;
        # Setup the resource/strain type/family structures
        T::Int,
        F::Int,
        M_per_type::Int=1,
        Mwaste::Int=0,
        S_per_family::Int=1,
        Sgen::Int=0,

        # This sets the cs
        muc::Float64=1.0,
        c0::Float64=0.0,
        c1::Float64=1.0,
        q::Float64=1.0,
        # This sets the Ds
        s::Float64=1.0,
        fs::Float64=0.45,
        fw::Float64=0.45,

        # These are simply distributions to use
        m=1.0, r=1.0, l=0.0,
        Ds=1e-10, Dr=1.0,

        # influx resource "natural" level
        food_R=0.0,
        usenthreads=nothing,
    )
        Ns = F * S_per_family + Sgen
        Nr = T * M_per_type + Mwaste

        # sort out the structure
        Sfamilies = fill(-1, Ns) # generalists by default
        for fi in 1:F
            for i in 1:S_per_family
                Sfamilies[(fi-1)*S_per_family+i] = fi
            end
        end
        Mtypes = fill(-1, Nr)
        for ti in 1:T
            for i in 1:M_per_type
                Mtypes[(ti-1)*M_per_type+i] = ti
            end
        end

        # Deal with the distributions
        if isa(m, Number)
            m = Dirac(m)
        elseif isa(m, Tuple) && length(m) == 2
            m = Normal(m[1], m[2])
        end
        if isa(r, Number)
            r = Dirac(r)
        elseif isa(r, Tuple) && length(r) == 2
            r = Normal(r[1], r[2])
        end
        if isa(l, Number)
            l = Dirac(l)
        elseif isa(l, Tuple) && length(l) == 2
            l = Normal(l[1], l[2])
        end
        if isa(Ds, Number)
            Ds = Dirac(Ds)
        elseif isa(Ds, Tuple) && length(Ds) == 2
            Ds = Normal(Ds[1], Ds[2])
        end
        if isa(Dr, Number)
            Dr = Dirac(Dr)
        elseif isa(Dr, Tuple) && length(Dr) == 2
            Dr = Normal(Dr[1], Dr[2])
        end

        new(
            Ns, Nr,
            T, M_per_type, Mwaste, F, S_per_family, Sgen,
            Sfamilies, Mtypes,
            muc, c0, c1, q,
            s, fs, fw,
            m, r, l,
            food_R,
            Ds, Dr,
            usenthreads
        )
    end
end
function (ms::AdaptedMarsland1)()
    g = fill(1.0, ms.Ns)
    w = fill(1.0, ms.Nr)

    m = clamp.(rand(ms.m, ms.Ns), 0.0, Inf)
    r = clamp.(rand(ms.r, ms.Nr), 0.0, Inf)

    # Generate c first
    c = fill(0.0, ms.Ns, ms.Nr)
    for i in 1:ms.Ns
        for a in 1:ms.Nr
            a_is_waste = ms.Mtypes[a] == -1
            if ms.Sfamilies[i] == -1 # this species is a generalist
                p = ms.muc / (ms.Nr * ms.c1)
            else
                if !a_is_waste && (ms.Sfamilies[i] == ms.Mtypes[a])
                    p = (ms.muc / (ms.Nr * ms.c1)) * (1 + ms.q * (ms.Nr - ms.M_per_type) / ms.M_per_type)
                else
                    p = (ms.muc / (ms.Nr * ms.c1)) * (1 - ms.q)
                end
            end
            X = rand(Bernoulli(p))
            c[i, a] = ms.c0 / ms.Nr + ms.c1 * X
        end
    end
    # And only fill l and D where there is a non-zero c
    l = fill(0.0, (ms.Ns, ms.Nr))
    D = fill(0.0, (ms.Ns, ms.Nr, ms.Nr))
    for i in 1:ms.Ns
        for a in 1:ms.Nr
            c_ = c[i, a]
            if !iszero(c_)
                l[i, a] = clamp(rand(ms.l), 0.0, 1.0)
                # D[i, :, a] = Dnoi[:, a]
                for b in 1:ms.Nr
                    ds = Float64[]
                    for g in 1:ms.Nr
                        fb = ms.Mtypes[b]
                        fg = ms.Mtypes[g]

                        d = if (fb != -1) && (fg == -1)
                            ms.fw / (ms.s * ms.Mwaste)
                        elseif (fb != -1) && (fb == fg)
                            ms.fs / (ms.s * ms.M_per_type)
                        elseif (fb != -1) && (fg != -1) && (fb != fg)
                            (1 - ms.fs - ms.fw) / (ms.s * (ms.Nr - ms.M_per_type - ms.Mwaste))
                        elseif (fb == -1) && (fg == -1)
                            (ms.fw + ms.fs) / (ms.s * ms.Mwaste)
                        elseif (fb == -1) && (fg != -1)
                            (1 - ms.fw - ms.fs) / (ms.s * (ms.Nr - ms.Mwaste))
                        else
                            throw(ErrorException("Unexpected case in type-family association"))
                        end

                        push!(ds, d)
                    end

                    D[i, :, b] .= rand(Dirichlet(ds))
                end
            end
        end
    end

    K = fill(0.0, ms.Nr)
    K[1] = ms.food_R / r[1]

    Ds = clamp.(rand(ms.Ds, ms.Ns), 0.0, Inf)
    Dr = clamp.(rand(ms.Dr, ms.Nr), 0.0, Inf)
    Ds = vcat(Ds, Dr)

    mmicrm_params = BMMiCRMParams(g, w, m, K, r, l, c, D, ms.usenthreads)
    BSMMiCRMParams(mmicrm_params, Ds)
end
export AdaptedMarsland1

################################################################################
# Others - mainly unstructured
################################################################################
function kaka(Ns, Nr;
    # the basic vars which are just all independent
    rdist=base10_lognormal(0.0, 0.0),
    mdist=base10_lognormal(0.0, 0.0),
    Dsdist=base10_lognormal(-12.0, 0.0),
    Drdist=base10_lognormal(0.0, 0.0),

    # for c, l and D
    sr=1.0,
    cdist=base10_lognormal(0.0, 0.0),
    ldist=Dirac(0.0),
    sb=1.0,

    # influx resource vars
    si=1.0,
    Kdist=Dirac(1.0)
)
    num_used_resources = clamp(round(Int, sr * Nr), 0, Nr)
    num_byproducts = clamp(round(Int, sb * Nr), 0, Nr)
    num_influx_resources = clamp(round(Int, si * Nr), 0, Nr)

    function ()
        g = fill(1.0, Ns)
        w = fill(1.0, Nr)

        # do the simple vars
        r = rand(rdist, Nr)
        m = rand(mdist, Ns)
        Ds = rand(Dsdist, Ns)
        Dr = rand(Drdist, Nr)

        # do c, l and D
        c = fill(0.0, Ns, Nr)
        l = fill(0.0, Ns, Nr)
        D = fill(0.0, Ns, Nr, Nr)
        for i in 1:Ns
            used_resources = sample(1:Nr, num_used_resources; replace=false)
            for a in used_resources
                c[i, a] = rand(cdist)
                l[i, a] = rand(ldist)

                byproducts = sample(1:Nr, num_byproducts; replace=false)
                bp_Ds = rand(length(byproducts))
                bp_Ds ./= sum(bp_Ds)
                for (b, byproduct_D) in zip(byproducts, bp_Ds)
                    D[i, b, a] = byproduct_D
                end
            end
        end

        # finally, influx resources

        mmicrm_params = BMMiCRMParams(g, w, m, K, r, l, c, D, rsg.usenthreads)
        BSMMiCRMParams(mmicrm_params, Ds)
    end
end


struct RSGJans3
    function RSGJans3(Ns, Nr;
    )
    end
end
function (rsg::RSGJans3)()
end
export RSGJans3
