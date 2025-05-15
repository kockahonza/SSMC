function random_matrix_initialization(S, M, c_sparsity=1.0, l_sparsity=0.35)
    # 110425 to start let us just randomly draw D and c without structure
    # similarly, we will generate some random guesses for the other parameters

    # constant dilution rate
    rnd = rand()
    r = fill(rnd, M)

    # universal death rate
    rnd2 = rand()
    m = rnd2

    # for simplicity, lets start with a single fed resource
    # chemostat feed rate 
    #K = fill(0.,M)
    #K[1] = 1.

    # lets allow resources some variability
    #K_dist = truncated(Normal(0.5,0.1), 0.0, 1.0)
    K_dist = Beta(0.1, 0.3)
    K = rand(K_dist, M)


    # leakage now. Lets assume its a pretty flat probability distribution
    leak = Beta(0.2 / l_sparsity, 0.2)
    l = rand(leak, (S, M))

    # most values around 0. This is essentially a proxy for sparsity
    c_i_alpha = Beta(0.5 / c_sparsity, 0.5)
    c = rand(c_i_alpha, (S, M))

    # finally, the most complicated distribution
    D = fill(0.0, (S, M, M))
    for i in 1:S
        for j in 1:M
            if c[i, j] > 0
                flag = true
                while flag
                    for k in 1:M
                        if j == k
                            D[i, k, j] = 0.0
                        else
                            D[i, k, j] = rand(Beta(0.5 / (M / 5), 0.5))
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


    Ds = fill(0.0, (S + M))
    Ds[1:S] .= 1e-5
    Ds[1+S] = 100
    Ds[S+2:S+M] .= 10

    return r, m, K, l, c, D, Ds
end


function BinaryRandomMatrix(a, b, p)
    # Generate a random binary matrix of size a x b with probability of 1 = p
    r = rand(a, b)
    m = fill(0, a, b)

    for i in 1:a
        for j in 1:b
            if r[i, j] < p
                m[i, j] = 1
            end
        end
    end

    return m
end

function marsland_initialization(S, M, SA=5, MA=5, q=0.9, c0=0.0, c1=1.0, muc=10,
    fs=0.45, fw=0.45, sparsity=0.2, l=0.8)
    # To start we are going to make a speciofic subset of assumptions available in the Marsland and Cui code.


    # let us begin by assigning the whole array c0
    c = fill(c0 / M, (S, M))

    # now we will calculate the block structure of the matrix
    F = ceil(M / MA) #number of resource classes
    T = ceil(S / SA) #number of species classes
    S_overlap = S % SA # number of species in the last class
    M_overlap = M % MA # number of resources in the last class
    # we will always assume that the last species class is the "general" class 
    println("T: ", T, " F: ", F)

    # we will sample the consumption matrix in block form

    for tt in 1:T
        for ff in 1:F
            if tt != T
                if ff == tt
                    p = muc / (M * c1) * (1 + q * (M - MA) / M)
                else
                    p = muc / (M * c1) * (1 - q)
                end

                if ff * MA > M
                    # ensure that the last block is not larger than the matrix
                    block = BinaryRandomMatrix(SA, M_overlap, p)
                    c[Int(1 + SA * (tt - 1)):Int(tt * SA), Int(1 + MA * (ff - 1)):Int(M)] .= block
                else
                    block = BinaryRandomMatrix(SA, MA, p)
                    c[Int(1 + SA * (tt - 1)):Int(tt * SA), Int(1 + MA * (ff - 1)):Int(ff * MA)] .= block
                end

            else
                # generalist class
                p = muc / (M * c1)
                if S_overlap != 0
                    block = BinaryRandomMatrix(S_overlap, M, p)
                    c[Int(1 + SA * (tt - 1)):Int(S), :] .= block
                else

                    block = BinaryRandomMatrix(SA, M, p)
                    c[Int(1 + SA * (tt - 1)):Int(tt * SA), :] .= block
                end
            end
        end
    end


    # Time for D_iab
    strain_class = 0
    D = fill(0.0, (S, M, M))
    for i in 1:S
        if (i - 1) % SA == 0
            strain_class += 1
            #println("Class: ", class, " i: ", i)
        end

        resource_class = 0
        for j in 1:M
            if (j - 1) % MA == 0
                resource_class += 1
            end

            #start with background levels
            bkg = (1 - fw - fs) / (M - MA)
            p = fill(bkg, M)

            if resource_class == strain_class
                if strain_class == T
                    if M_overlap != 0
                        p[(M-M_overlap):M] .= fw + fs
                    else
                        p[(M-MA):M] .= fw + fs
                    end
                else
                    #the within class resource
                    upper_limit = minimum(((strain_class - 1) * MA + MA, M))
                    p[1+(strain_class-1)*MA:upper_limit] .= fs

                    # the waste resources
                    if M_overlap != 0
                        p[(M-M_overlap):M] .= fw
                    else
                        p[(M-MA):M] .= fw
                    end
                end
            else
                p = fill(1.0, M)
            end



            #lets sample the distribution
            vec = rand(Dirichlet(p))
            D[i, :, j] = vec

        end
    end


    # constant dilution rate
    rnd = rand()
    r = fill(rnd, M)

    # universal death rate
    rnd2 = rand()
    m = rnd2

    # for simplicity, lets start with a single fed resource
    # chemostat feed rate 
    #K = fill(0.,M)
    #K[1] = 1.

    # lets allow resources some variability
    #K_dist = truncated(Normal(0.5,0.1), 0.0, 1.0)
    K_dist = Beta(0.1, 0.3)
    K = rand(K_dist, M)


    # leakage now. Lets assume its a pretty flat probability distribution
    leak = Beta(0.2 / sparsity, 0.2)
    l = rand(leak, (S, M))


    Ds = fill(0.0, (S + M))
    Ds[1:S] .= 1e-5
    Ds[1+S] = 100
    Ds[S+2:S+M] .= 10

    return r, m, K, l, c, D, Ds

end
