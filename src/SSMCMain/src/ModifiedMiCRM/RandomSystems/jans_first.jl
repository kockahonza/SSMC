"""
Simple, almost entirely unstructured sampling method. Has an inherent
strain-resource sparsity of num_used_resources / Nr and an inherent
resource-resource sparsity of num_byproducts / Nr.
The only structure comes from each resource being either an "influx"
one which has a non-zero K and hence is being added to the system
or not an "influx" resource which are exclusively added through being
byproducts of consumption processes.
"""
struct RSGJans1{Dm,Dr,DDs,DDr,Di,DK,Ds1,Ds2,Dc,Dl,Dc2,Dl2}
    Ns::Int # number of strains
    Nr::Int # number of resources

    m::Dm   # upkeep energy rate distribution
    r::Dr   # resource dilution rate distribution
    Ds::DDs # strain diffusion constant distribution
    Dr::DDr # resource diffusion constant distribution

    num_influx_resources::Di # a discrete distribution for generating the number of resources with non-zero K
    K::DK # distribution of Ks for those resources which are being added

    num_used_resources::Ds1 # discrete distribution of the number of resources each strain eats
    num_byproducts::Ds2     # discrete distribution of the number of byproducts for each consumption process

    c::Dc # distribution for those consumption rates which are not zero
    l::Dl # leakage distribution

    cinflux::Dc2
    linflux::Dl2

    usenthreads::Union{Nothing,Int}
    function RSGJans1(Ns, Nr;
        m=1.0, r=1.0, Ds=1.0, Dr=1.0,
        num_influx_resources=nothing, Kp=nothing, sparsity_influx=nothing, K=1.0,
        num_used_resources=nothing, sparsity_resources=nothing,
        num_byproducts=nothing, sparsity_byproducts=nothing,
        c=1.0, l=0.5,
        cinflux=c, linflux=l,
        usenthreads=nothing
    )
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
        if isa(K, Number)
            K = Dirac(K)
        elseif isa(K, Tuple) && length(K) == 2
            K = Normal(K[1], K[2])
        end
        if isa(c, Number)
            c = Dirac(c)
        elseif isa(c, Tuple) && length(c) == 2
            c = Normal(c[1], c[2])
        end
        if isa(l, Number)
            l = Dirac(l)
        elseif isa(l, Tuple) && length(l) == 2
            l = Normal(l[1], l[2])
        end
        if isnothing(cinflux)
            cinflux = c
        elseif isa(cinflux, Number)
            cinflux = Dirac(cinflux)
        elseif isa(cinflux, Tuple) && length(cinflux) == 2
            cinflux = Normal(cinflux[1], cinflux[2])
        end
        if isnothing(linflux)
            linflux = l
        elseif isa(linflux, Number)
            linflux = Dirac(linflux)
        elseif isa(linflux, Tuple) && length(linflux) == 2
            linflux = Normal(linflux[1], linflux[2])
        end

        if isnothing(num_influx_resources) && isnothing(Kp) && isnothing(sparsity_influx)
            num_influx_resources = Dirac(Nr)
        elseif isnothing(num_influx_resources)
            if isnothing(sparsity_influx)
                num_influx_resources = Binomial(Nr, Kp)
            else
                num_influx_resources = Dirac(round(Int, sparsity_influx * Nr))
            end
        elseif !isnothing(Kp) && !isnothing(sparsity_influx)
            @warn "RSGJans1 has been passed multiple kwargs for influx resources some are being ignored"
        end

        if isnothing(num_used_resources) && isnothing(sparsity_resources)
            num_used_resources = Nr
        elseif isnothing(num_used_resources)
            num_used_resources = round(Int, sparsity_resources * Nr)
        end
        if isa(num_used_resources, Number)
            num_used_resources = Dirac(num_used_resources)
        end

        if isnothing(num_byproducts) && isnothing(sparsity_byproducts)
            num_byproducts = Nr
        elseif isnothing(num_byproducts)
            num_byproducts = round(Int, sparsity_byproducts * Nr)
        end
        if isa(num_byproducts, Number)
            num_byproducts = Dirac(num_byproducts)
        end

        new{
            typeof(m),typeof(r),typeof(Ds),typeof(Dr),typeof(num_influx_resources),typeof(K),
            typeof(num_used_resources),typeof(num_byproducts),typeof(c),typeof(l),typeof(cinflux),typeof(linflux)
        }(Ns, Nr, m, r, Ds, Dr, num_influx_resources, K, num_used_resources, num_byproducts, c, l, cinflux, linflux, usenthreads)
    end
end
function (rsg::RSGJans1)()
    # as usual
    g = fill(1.0, rsg.Ns)
    w = fill(1.0, rsg.Nr)

    m = clamp.(rand(rsg.m, rsg.Ns), 0.0, Inf)
    r = clamp.(rand(rsg.r, rsg.Nr), 0.0, Inf)

    K = fill(0.0, rsg.Nr)
    num_influx_resources = rand(rsg.num_influx_resources)
    influx_resources = sample(1:rsg.Nr, num_influx_resources; replace=false)
    for a in influx_resources
        K[a] = clamp(rand(rsg.K), 0.0, Inf)
    end

    l = fill(0.0, (rsg.Ns, rsg.Nr))
    c = fill(0.0, (rsg.Ns, rsg.Nr))
    D = fill(0.0, (rsg.Ns, rsg.Nr, rsg.Nr))
    for i in 1:rsg.Ns
        num_used_resources = rand(rsg.num_used_resources)
        used_resources = sample(1:rsg.Nr, num_used_resources; replace=false)
        for cr in used_resources
            csampler = iszero(K[cr]) ? rsg.c : rsg.cinflux
            lsampler = iszero(K[cr]) ? rsg.l : rsg.linflux
            c[i, cr] = clamp(rand(csampler), 0.0, Inf)
            l[i, cr] = clamp(rand(lsampler), 0.0, 1.0)

            num_byproducts = rand(rsg.num_byproducts)
            byproducts = sample(1:rsg.Nr, num_byproducts; replace=false)
            bp_Ds = rand(length(byproducts))
            bp_Ds ./= sum(bp_Ds)
            for (bp, bp_D) in zip(byproducts, bp_Ds)
                D[i, bp, cr] = bp_D
            end
        end
    end

    Ds = clamp.(rand(rsg.Ds, rsg.Ns), 0.0, Inf)
    Dr = clamp.(rand(rsg.Dr, rsg.Nr), 0.0, Inf)
    Ds = vcat(Ds, Dr)

    mmicrm_params = BMMiCRMParams(g, w, m, K, r, l, c, D, rsg.usenthreads)
    BSMMiCRMParams(mmicrm_params, Ds)
end
export RSGJans1

"""
Modified version of steven's Marsland sampler
"""
struct Marsland2
    Ns::Int
    Nr::Int
    SA::Int
    MA::Int
    q::Float64
    c0::Float64
    c1::Float64
    muc::Float64
    fs::Float64
    fw::Float64
    sparsity::Float64
    function Marsland2(Ns, Nr;
        SA=5, MA=5,
        q=0.9, c0=0.0, c1=1.0,
        muc=10, fs=0.45, fw=0.45,
        sparsity=0.2
    )
        new(Ns, Nr, SA, MA, q, c0, c1, muc, fs, fw, sparsity)
    end
end
function (ms::Marsland2)()
    # let us begin by assigning the whole array c0
    c = fill(ms.c0 / ms.Nr, (ms.Ns, ms.Nr))

    # now we will calculate the block structure of the matrix
    F = ceil(ms.Nr / ms.MA) #number of resource classes
    T = ceil(ms.Ns / ms.SA) #number of species classes
    S_overlap = ms.Ns % ms.SA # number of species in the last class
    M_overlap = ms.Nr % ms.MA # number of resources in the last class
    # we will always assume that the last species class is the "general" class 
    # println("T: ", T, " F: ", F)

    # we will sample the consumption matrix in block form

    for tt in 1:T
        for ff in 1:F
            if tt != T
                if ff == tt
                    p = ms.muc / (ms.Nr * ms.c1) * (1 + ms.q * (ms.Nr - ms.MA) / ms.Nr)
                else
                    p = ms.muc / (ms.Nr * ms.c1) * (1 - ms.q)
                end

                # FIX: This is obviously wrong and urgently needs a fix!
                if p > 1.0
                    p = 1.0
                end

                if ff * ms.MA > ms.Nr
                    # ensure that the last block is not larger than the matrix
                    block = random_binary_matrix(ms.SA, M_overlap, p)
                    c[Int(1 + ms.SA * (tt - 1)):Int(tt * ms.SA), Int(1 + ms.MA * (ff - 1)):Int(ms.Nr)] .= block
                else
                    block = random_binary_matrix(ms.SA, ms.MA, p)
                    c[Int(1 + ms.SA * (tt - 1)):Int(tt * ms.SA), Int(1 + ms.MA * (ff - 1)):Int(ff * ms.MA)] .= block
                end

            else
                # generalist class
                p = ms.muc / (ms.Nr * ms.c1)

                # FIX: This is obviously wrong and urgently needs a fix!
                if p > 1.0
                    p = 1.0
                end

                if S_overlap != 0
                    block = random_binary_matrix(S_overlap, ms.Nr, p)
                    c[Int(1 + ms.SA * (tt - 1)):Int(ms.Ns), :] .= block
                else

                    block = random_binary_matrix(ms.SA, ms.Nr, p)
                    c[Int(1 + ms.SA * (tt - 1)):Int(tt * ms.SA), :] .= block
                end
            end
        end
    end


    # Time for D_iab
    strain_class = 0
    D = fill(0.0, (ms.Ns, ms.Nr, ms.Nr))
    for i in 1:ms.Ns
        if (i - 1) % ms.SA == 0
            strain_class += 1
            #println("Class: ", class, " i: ", i)
        end

        resource_class = 0
        for j in 1:ms.Nr
            if (j - 1) % ms.MA == 0
                resource_class += 1
            end

            #start with background levels
            bkg = (1 - ms.fw - ms.fs) / (ms.Nr - ms.MA)
            p = fill(bkg, ms.Nr)

            if resource_class == strain_class
                if strain_class == T
                    if M_overlap != 0
                        p[(ms.Nr-M_overlap):ms.Nr] .= ms.fw + ms.fs
                    else
                        p[(ms.Nr-ms.MA):ms.Nr] .= ms.fw + ms.fs
                    end
                else
                    #the within class resource
                    upper_limit = minimum(((strain_class - 1) * ms.MA + ms.MA, ms.Nr))
                    p[1+(strain_class-1)*ms.MA:upper_limit] .= ms.fs

                    # the waste resources
                    if M_overlap != 0
                        p[(ms.Nr-M_overlap):ms.Nr] .= ms.fw
                    else
                        p[(ms.Nr-ms.MA):ms.Nr] .= ms.fw
                    end
                end
            else
                p = fill(1.0, ms.M)
            end



            #lets sample the distribution
            vec = rand(Dirichlet(p))
            D[i, :, j] = vec

        end
    end


    # constant dilution rate
    r = fill(rand(), ms.M)

    # universal death rate
    m = fill(rand(), ms.Ns)

    # for simplicity, lets start with a single fed resource
    # chemostat feed rate 
    #K = fill(0.,ms.M)
    #K[1] = 1.

    # lets allow resources some variability
    #K_dist = truncated(Normal(0.5,0.1), 0.0, 1.0)
    K_dist = Beta(0.1, 0.3)
    K = rand(K_dist, ms.M)


    # leakage now. Lets assume its a pretty flat probability distribution
    leak = Beta(0.2 / ms.sparsity, 0.2)
    l = rand(leak, (ms.Ns, ms.M))


    Ds = fill(0.0, (ms.Ns + ms.M))
    Ds[1:ms.Ns] .= 1e-5
    Ds[1+ms.Ns] = 100
    Ds[ms.Ns+2:ms.Ns+ms.M] .= 10

    g = fill(1.0, ms.Ns)
    w = fill(1.0, ms.M)

    p = BMMiCRMParams(g, w, m, K, r, l, c, D, nothing)
    BSMMiCRMParams(p, Ds)
end
export Marsland2

################################################################################
# New stuff
################################################################################
function get_random_metabolic_process(prob_eating, cdist, ldist, num_byproducts, Nr)
    if rand() < prob_eating
        c = clamp(rand(cdist), 0.0, Inf)
        l = clamp(rand(ldist), 0.0, 1.0)
        byproducts = sample(1:Nr, num_byproducts; replace=false)
        bp_Ds = rand(length(byproducts))
        bp_Ds ./= sum(bp_Ds)
        D = fill(0.0, Nr)
        for (bp, bp_D) in zip(byproducts, bp_Ds)
            D[bp] = bp_D
        end

        c, l, D
    else
        0.0, 0.0, fill(0.0, Nr)
    end
end

struct JansSampler3
    Ns::Int # number of strains
    Nr::Int # number of resources

    m::Distribution
    r::Distribution

    num_influx_resources::Distribution
    K::Distribution

    prob_eating::Float64
    prob_eating_influx::Float64
    num_byproducts::Distribution

    c::Distribution
    l::Distribution
    cinflux::Distribution
    linflux::Distribution

    Ds::Distribution
    Dr::Distribution
    Drinflux::Distribution

    usenthreads::Union{Nothing,Int}
    function JansSampler3(Ns, Nr;
        m=1.0, r=1.0,
        num_influx_resources=1, K=1.0,
        prob_eating=1.0, prob_eating_influx=1.0, num_byproducts=0,
        c=1.0, l=0.5, cinflux=c, linflux=l,
        Ds=1e-12, Dr=1.0, Drinflux=Dr,
        usenthreads=nothing
    )
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
        if isa(num_influx_resources, Integer)
            num_influx_resources = Dirac(num_influx_resources)
        end
        if isa(K, Number)
            K = Dirac(K)
        elseif isa(K, Tuple) && length(K) == 2
            K = Normal(K[1], K[2])
        end
        if isa(num_byproducts, Integer)
            num_byproducts = Dirac(num_byproducts)
        end
        if isa(c, Number)
            c = Dirac(c)
        elseif isa(c, Tuple) && length(c) == 2
            c = Normal(c[1], c[2])
        end
        if isa(l, Number)
            l = Dirac(l)
        elseif isa(l, Tuple) && length(l) == 2
            l = Normal(l[1], l[2])
        end
        if isnothing(cinflux)
            cinflux = c
        elseif isa(cinflux, Number)
            cinflux = Dirac(cinflux)
        elseif isa(cinflux, Tuple) && length(cinflux) == 2
            cinflux = Normal(cinflux[1], cinflux[2])
        end
        if isnothing(linflux)
            linflux = l
        elseif isa(linflux, Number)
            linflux = Dirac(linflux)
        elseif isa(linflux, Tuple) && length(linflux) == 2
            linflux = Normal(linflux[1], linflux[2])
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
        if isnothing(Drinflux)
            Drinflux = Dr
        elseif isa(Drinflux, Number)
            Drinflux = Dirac(Drinflux)
        elseif isa(Drinflux, Tuple) && length(Drinflux) == 2
            Drinflux = Normal(Drinflux[1], Drinflux[2])
        end

        new(Ns, Nr, m, r, num_influx_resources, K, prob_eating, prob_eating_influx, num_byproducts, c, l, cinflux, linflux, Ds, Dr, Drinflux, usenthreads)
    end
end
function (rsg::JansSampler3)()
    # as usual
    g = fill(1.0, rsg.Ns)
    w = fill(1.0, rsg.Nr)

    m = clamp.(rand(rsg.m, rsg.Ns), 0.0, Inf)
    r = clamp.(rand(rsg.r, rsg.Nr), 0.0, Inf)

    K = fill(0.0, rsg.Nr)
    num_influx_resources = rand(rsg.num_influx_resources)
    influx_resources = sample(1:rsg.Nr, num_influx_resources; replace=false)
    for a in influx_resources
        K[a] = clamp(rand(rsg.K), 0.0, Inf)
    end

    l = fill(0.0, (rsg.Ns, rsg.Nr))
    c = fill(0.0, (rsg.Ns, rsg.Nr))
    D = fill(0.0, (rsg.Ns, rsg.Nr, rsg.Nr))
    for a in 1:rsg.Nr
        isinflux = !iszero(K[a])
        for i in 1:rsg.Ns
            metabolic_process = if !isinflux
                get_random_metabolic_process(rsg.prob_eating, rsg.c, rsg.l, rand(rsg.num_byproducts), rsg.Nr)
            else
                get_random_metabolic_process(rsg.prob_eating_influx, rsg.cinflux, rsg.linflux, rand(rsg.num_byproducts), rsg.Nr)
            end
            c[i, a] = metabolic_process[1]
            l[i, a] = metabolic_process[2]
            D[i, :, a] .= metabolic_process[3]
        end
    end

    Ds = clamp.(rand(rsg.Ds, rsg.Ns), 0.0, Inf)
    Dr = clamp.(rand(rsg.Dr, rsg.Nr), 0.0, Inf)
    for a in influx_resources
        Dr[a] = clamp(rand(rsg.Drinflux), 0.0, Inf)
    end
    Ds = vcat(Ds, Dr)

    mmicrm_params = BMMiCRMParams(g, w, m, K, r, l, c, D, rsg.usenthreads)
    BSMMiCRMParams(mmicrm_params, Ds)
end
export JansSampler3

function single_influx_resource_sampler(Ns, Nr;
    kwargs...
)
    rsg = RSGJans1(Ns, Nr; kwargs...)
    function ()
        ps = rsg()
    end
end
