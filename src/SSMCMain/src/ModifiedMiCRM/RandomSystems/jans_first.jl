"""
Simple, almost entirely unstructured sampling method. Has an inherent
strain-resource sparsity of num_used_resources / Nr and an inherent
resource-resource sparsity of num_byproducts / Nr.
The only structure comes from each resource being either an "influx"
one which has a non-zero K and hence is being added to the system
or not an "influx" resource which are exclusively added through being
byproducts of consumption processes.
"""
struct RSGJans1{Dm,Dr,DDN,DDR,DK,Ds1,Ds2,Dc,Dl}
    Ns::Int # number of strains
    Nr::Int # number of resources

    m::Dm   # upkeep energy rate distribution
    r::Dr   # resource dilution rate distribution
    DS::DDN # strain diffusion constant distribution
    DR::DDR # resource diffusion constant distribution

    Kp::Float64 # probability of a resource being added to the system
    K::DK       # distribution of Ks for those resources which are being added

    num_used_resources::Ds1 # discrete distribution of the number of resources each strain eats
    num_byproducts::Ds2     # discrete distribution of the number of byproducts for each consumption process

    c::Dc # distribution for those consumption rates which are not zero
    l::Dl # leakage distribution

    usenthreads::Union{Nothing,Int}
    function RSGJans1(Ns, Nr;
        m=1.0, r=1.0, DS=1.0, DR=1.0,
        Kp=1.0, K=1.0,
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
        if isa(DS, Number)
            DS = Dirac(DS)
        elseif isa(DS, Tuple) && length(DS) == 2
            DS = Normal(DS[1], DS[2])
        end
        if isa(DR, Number)
            DR = Dirac(DR)
        elseif isa(DR, Tuple) && length(DR) == 2
            DR = Normal(DR[1], DR[2])
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
            typeof(m),typeof(r),typeof(DS),typeof(DR),typeof(K),
            typeof(num_used_resources),typeof(num_byproducts),typeof(c),typeof(l)
        }(Ns, Nr, m, r, DS, DR, Kp, K, num_used_resources, num_byproducts, c, l, usenthreads)
    end
end
function (rsg::RSGJans1)()
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

    DS = rand(rsg.DS, rsg.Ns)
    DR = rand(rsg.DR, rsg.Nr)
    # DR = fill(0.0, rsg.Nr)
    # for a in 1:rsg.Nr
    #     if K[a] != 0.0
    #         DR[a] = rand(rsg.DR)
    #     end
    # end
    Ds = vcat(DS, DR)

    mmicrm_params = BMMiCRMParams(g, w, m, K, r, l, c, D, rsg.usenthreads)
    BSMMiCRMParams(mmicrm_params, Ds)
end
export RSGJans1

"""
Here we distinguish between the influx and not-influx resources even more
"""
struct RSGJans2{Dm,Dr,DDN,DDR,DK,Dci,Dli,Dcb,Dlb}
    Ns::Int
    Nr::Int

    m::Dm
    r::Dr
    DS::DDN
    DR::DDR

    Kp::Float64
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
        m=1.0, r=1.0, DS=1.0, DR=1.0,
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
        if isa(DS, Number)
            DS = Dirac(DS)
        elseif isa(DS, Tuple) && length(DS) == 2
            DS = Normal(DS[1], DS[2])
        end
        if isa(DR, Number)
            DR = Dirac(DR)
        elseif isa(DR, Tuple) && length(DR) == 2
            DR = Normal(DR[1], DR[2])
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
            typeof(m),typeof(r),typeof(DS),typeof(DR),typeof(K),
            typeof(c_in),typeof(l_in),typeof(c_bp),typeof(l_bp)
        }(Ns, Nr, m, r, DS, DR, Kp, K,
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

    DS = rand(rsg.DS, rsg.Ns)
    DR = rand(rsg.DR, rsg.Nr)
    # DR = fill(0.0, rsg.Nr)
    # for a in 1:rsg.Nr
    #     if K[a] != 0.0
    #         DR[a] = rand(rsg.DR)
    #     end
    # end
    Ds = vcat(DS, DR)

    mmicrm_params = BMMiCRMParams(g, w, m, K, r, l, c, D, rsg.usenthreads)
    BSMMiCRMParams(mmicrm_params, Ds)
end
export RSGJans2
