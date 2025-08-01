"""
Simple, almost entirely unstructured sampling method. Has an inherent
strain-resource sparsity of num_used_resources / Nr and an inherent
resource-resource sparsity of num_byproducts / Nr.
The only structure comes from each resource being either an "influx"
one which has a non-zero K and hence is being added to the system
or not an "influx" resource which are exclusively added through being
byproducts of consumption processes.
"""
struct RSGJans1{Dm,Dr,DDs,DDr,Di,DK,Ds1,Ds2,Dc,Dl}
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

    usenthreads::Union{Nothing,Int}
    function RSGJans1(Ns, Nr;
        m=1.0, r=1.0, Ds=1.0, Dr=1.0,
        num_influx_resources=nothing, Kp=nothing, sparsity_influx=nothing, K=1.0,
        num_used_resources=nothing, sparsity_resources=nothing,
        num_byproducts=nothing, sparsity_byproducts=nothing,
        c=1.0, l=0.5,
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
            typeof(num_used_resources),typeof(num_byproducts),typeof(c),typeof(l)
        }(Ns, Nr, m, r, Ds, Dr, num_influx_resources, K, num_used_resources, num_byproducts, c, l, usenthreads)
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
            c[i, cr] = clamp(rand(rsg.c), 0.0, Inf)
            l[i, cr] = clamp(rand(rsg.l), 0.0, 1.0)

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
Here we distinguish between the influx and not-influx resources even more
"""
struct RSGJans2{Dm,Dr,DDs,DDr,DK,Dci,Dli,Dcb,Dlb}
    Ns::Int
    Nr::Int

    m::Dm
    r::Dr
    Ds::DDs
    Dr::DDr

    Kp::Float64 # FIX: adapt to copy the behaviour of RSGJans1
    K::DK

    num_used_in_resources::Int
    num_in_byproducts::Int
    c_in::Dci
    l_in::Dli

    num_used_bp_resources::Int
    num_bp_byproducts::Int
    c_bp::Dcb
    l_bp::Dlb

    usenthreads::Union{Nothing,Int}
    function RSGJans2(Ns, Nr;
        m=1.0, r=1.0, Ds=1.0, Dr=1.0,
        Kp=1.0, K=1.0,
        sparsity_in_resources=1.0, sparsity_in_byproducts=1.0,
        c_in=1.0, l_in=0.5,
        sparsity_bp_resources=1.0, sparsity_bp_byproducts=1.0,
        c_bp=0.0, l_bp=0.0,
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

        num_used_in_resources = round(Int, sparsity_in_resources * Nr)
        num_in_byproducts = round(Int, sparsity_in_byproducts * Nr)
        if isa(c_in, Number)
            c_in = Dirac(c_in)
        elseif isa(c_in, Tuple) && length(c_in) == 2
            c_in = Normal(c_in[1], c_in[2])
        end
        if isa(l_in, Number)
            l_in = Dirac(l_in)
        elseif isa(l_in, Tuple) && length(l_in) == 2
            l_in = Normal(l_in[1], l_in[2])
        end

        num_used_bp_resources = round(Int, sparsity_bp_resources * Nr)
        num_bp_byproducts = round(Int, sparsity_bp_byproducts * Nr)
        if isa(c_bp, Number)
            c_bp = Dirac(c_bp)
        elseif isa(c_bp, Tuple) && length(c_bp) == 2
            c_bp = Normal(c_bp[1], c_bp[2])
        end
        if isa(l_bp, Number)
            l_bp = Dirac(l_bp)
        elseif isa(l_bp, Tuple) && length(l_bp) == 2
            l_bp = Normal(l_bp[1], l_bp[2])
        end

        new{
            typeof(m),typeof(r),typeof(Ds),typeof(Dr),typeof(K),
            typeof(c_in),typeof(l_in),typeof(c_bp),typeof(l_bp)
        }(Ns, Nr, m, r, Ds, Dr, Kp, K,
            num_used_in_resources, num_in_byproducts, c_in, l_in,
            num_used_bp_resources, num_bp_byproducts, c_bp, l_bp,
            usenthreads
        )
    end
end
function (rsg::RSGJans2)()
    # as usual
    g = fill(1.0, rsg.Ns)
    w = fill(1.0, rsg.Nr)

    m = rand(rsg.m, rsg.Ns)
    r = rand(rsg.r, rsg.Nr)

    K = fill(0.0, rsg.Nr)
    for a in 1:rsg.Nr
        if rand() < rsg.Kp
            K[a] = rand(rsg.K)
        end
    end

    l = fill(0.0, (rsg.Ns, rsg.Nr))
    c = fill(0.0, (rsg.Ns, rsg.Nr))
    D = fill(0.0, (rsg.Ns, rsg.Nr, rsg.Nr))
    for i in 1:rsg.Ns
        num_resources = rand(rsg.num_used_resources)
        consumed_resources = sample(1:rsg.Nr, num_resources; replace=false)
        for cr in consumed_resources
            c[i, cr] = abs(rand(rsg.c))
            l[i, cr] = rand(rsg.l)

            num_byproducts = rand(rsg.num_byproducts)
            byproducts = sample(1:rsg.Nr, num_byproducts; replace=false)
            bp_Ds = rand(length(byproducts))
            bp_Ds ./= sum(bp_Ds)
            for (bp, bp_D) in zip(byproducts, bp_Ds)
                D[i, bp, cr] = bp_D
            end
        end
    end

    Ds = rand(rsg.Ds, rsg.Ns)
    Dr = rand(rsg.Dr, rsg.Nr)
    # Dr = fill(0.0, rsg.Nr)
    # for a in 1:rsg.Nr
    #     if K[a] != 0.0
    #         Dr[a] = rand(rsg.Dr)
    #     end
    # end
    Ds = vcat(Ds, Dr)

    mmicrm_params = BMMiCRMParams(g, w, m, K, r, l, c, D, rsg.usenthreads)
    BSMMiCRMParams(mmicrm_params, Ds)
end
export RSGJans2

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
