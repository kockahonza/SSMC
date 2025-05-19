################################################################################
# Util and scratch
################################################################################
function random_binary_matrix(a, b, p)
    Int.(rand(Bernoulli(p), a, b))
end

################################################################################
# Generators
################################################################################
struct RSGStevens1
    Ns::Int
    Nr::Int
    c_sparsity::Float64
    l_sparsity::Float64
    r
    K
    usenthreads::Union{Nothing,Int}
    function RSGStevens1(Ns, Nr, c_sparsity, l_sparsity, r = nothing, K=nothing, usenthreads=nothing)
        new(Ns, Nr, c_sparsity, l_sparsity, r, K, usenthreads)
    end
end
function (rsg::RSGStevens1)()
    # as usual
    g = fill(1.0, rsg.Ns)
    w = fill(1.0, rsg.Nr)

    # sample death rates
    m_dist = Normal(0.5, 0.5)
    m = rand(m_dist, rsg.Ns)
    m = abs.(m)
    #m = fill(rand(), rsg.Ns)

    # for simplicity, lets start with a single fed resource
    # chemostat feed rate 
    #K = fill(0.,M)
    #K[1] = 1.

    # lets allow resources some variability
    #K_dist = truncated(Normal(0.5,0.1), 0.0, 1.0)
    if isnothing(rsg.K)
        K_dist = Beta(0.1, 0.3)
        K = rand(K_dist, rsg.Nr)
    else
        K = rsg.K
    end


    # constant dilution rate
    if isnothing(rsg.r)
        r = fill(rand(), rsg.Nr)
    else
        r = rsg.r
    end

    # leakage now. Lets assume its a pretty flat probability distribution
    leak = Beta(0.2 / rsg.l_sparsity, 0.2)
    l = rand(leak, (rsg.Ns, rsg.Nr))

    # most values around 0. This is essentially a proxy for sparsity
    c_i_alpha = Beta(0.5 / rsg.c_sparsity, 0.5)
    c = rand(c_i_alpha, (rsg.Ns, rsg.Nr))

    # finally, the most complicated distribution
    D = fill(0.0, (rsg.Ns, rsg.Nr, rsg.Nr))
    for i in 1:rsg.Ns
        for j in 1:rsg.Nr
            if c[i, j] > 0
                flag = true
                while flag
                    for k in 1:rsg.Nr
                        if j == k
                            D[i, k, j] = 0.0
                        else
                            D[i, k, j] = rand(Beta(0.5 / (rsg.Nr / 5), 0.5))
                        end
                    end
                    # check if the sum of the row is less than 1
                    if sum(D[i, :, j]) < 1.0
                        flag = false
                    end
                end
            end

        end
    end

    Ds = fill(0.0, (rsg.Ns + rsg.Nr))
    Ds[1:rsg.Ns] .= 1e-5
    Ds[1+rsg.Ns] = 100
    Ds[rsg.Ns+2:rsg.Ns+rsg.Nr] .= 10

    # return a spatial mmicrm params struct without a concrete space and with threading on the nospace level
    p = BMMiCRMParams(g, w, m, K, r, l, c, D, rsg.usenthreads)
    BSMMiCRMParams(p, Ds)
end
export RSGStevens1

"""
Stevens first version of the marsland sampler, completely unmodified just restructured
"""
struct MarslandSampler1
    S::Int
    M::Int
    SA::Int
    MA::Int
    q::Float64
    c0::Float64
    c1::Float64
    muc::Float64
    fs::Float64
    fw::Float64
    sparsity::Float64
    function MarslandSampler1(S, M;
        SA=5, MA=5,
        q=0.9, c0=0.0, c1=1.0,
        muc=10, fs=0.45, fw=0.45,
        sparsity=0.2
    )
        new(S, M, SA, MA, q, c0, c1, muc, fs, fw, sparsity)
    end
end
function (ms::MarslandSampler1)()
    # let us begin by assigning the whole array c0
    c = fill(ms.c0 / ms.M, (ms.S, ms.M))

    # now we will calculate the block structure of the matrix
    F = ceil(ms.M / ms.MA) #number of resource classes
    T = ceil(ms.S / ms.SA) #number of species classes
    S_overlap = ms.S % ms.SA # number of species in the last class
    M_overlap = ms.M % ms.MA # number of resources in the last class
    # we will always assume that the last species class is the "general" class 
    # println("T: ", T, " F: ", F)

    # we will sample the consumption matrix in block form

    for tt in 1:T
        for ff in 1:F
            if tt != T
                if ff == tt
                    p = ms.muc / (ms.M * ms.c1) * (1 + ms.q * (ms.M - ms.MA) / ms.M)
                else
                    p = ms.muc / (ms.M * ms.c1) * (1 - ms.q)
                end

                # FIX: This is obviously wrong and urgently needs a fix!
                if p > 1.0
                    p = 1.0
                end

                if ff * ms.MA > ms.M
                    # ensure that the last block is not larger than the matrix
                    block = random_binary_matrix(ms.SA, M_overlap, p)
                    c[Int(1 + ms.SA * (tt - 1)):Int(tt * ms.SA), Int(1 + ms.MA * (ff - 1)):Int(ms.M)] .= block
                else
                    block = random_binary_matrix(ms.SA, ms.MA, p)
                    c[Int(1 + ms.SA * (tt - 1)):Int(tt * ms.SA), Int(1 + ms.MA * (ff - 1)):Int(ff * ms.MA)] .= block
                end

            else
                # generalist class
                p = ms.muc / (ms.M * ms.c1)

                # FIX: This is obviously wrong and urgently needs a fix!
                if p > 1.0
                    p = 1.0
                end

                if S_overlap != 0
                    block = random_binary_matrix(S_overlap, ms.M, p)
                    c[Int(1 + ms.SA * (tt - 1)):Int(ms.S), :] .= block
                else

                    block = random_binary_matrix(ms.SA, ms.M, p)
                    c[Int(1 + ms.SA * (tt - 1)):Int(tt * ms.SA), :] .= block
                end
            end
        end
    end


    # Time for D_iab
    strain_class = 0
    D = fill(0.0, (ms.S, ms.M, ms.M))
    for i in 1:ms.S
        if (i - 1) % ms.SA == 0
            strain_class += 1
            #println("Class: ", class, " i: ", i)
        end

        resource_class = 0
        for j in 1:ms.M
            if (j - 1) % ms.MA == 0
                resource_class += 1
            end

            #start with background levels
            bkg = (1 - ms.fw - ms.fs) / (ms.M - ms.MA)
            p = fill(bkg, ms.M)

            if resource_class == strain_class
                if strain_class == T
                    if M_overlap != 0
                        p[(ms.M-M_overlap):ms.M] .= ms.fw + ms.fs
                    else
                        p[(ms.M-ms.MA):ms.M] .= ms.fw + ms.fs
                    end
                else
                    #the within class resource
                    upper_limit = minimum(((strain_class - 1) * ms.MA + ms.MA, ms.M))
                    p[1+(strain_class-1)*ms.MA:upper_limit] .= ms.fs

                    # the waste resources
                    if M_overlap != 0
                        p[(ms.M-M_overlap):ms.M] .= ms.fw
                    else
                        p[(ms.M-ms.MA):ms.M] .= ms.fw
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
    m = fill(rand(), ms.S)

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
    l = rand(leak, (ms.S, ms.M))


    Ds = fill(0.0, (ms.S + ms.M))
    Ds[1:ms.S] .= 1e-5
    Ds[1+ms.S] = 100
    Ds[ms.S+2:ms.S+ms.M] .= 10

    g = fill(1.0, ms.S)
    w = fill(1.0, ms.M)

    p = BMMiCRMParams(g, w, m, K, r, l, c, D, nothing)
    BSMMiCRMParams(p, Ds)
end
export MarslandSampler1
