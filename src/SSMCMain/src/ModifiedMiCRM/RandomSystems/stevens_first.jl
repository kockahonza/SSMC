"""
110425 to start let us just randomly draw D and c without structure
similarly, we will generate some random guesses for the other parameters
"""
function rg_stevens1_original(Ns, Nr, c_sparsity=1.0, l_sparsity=0.35)
    # constant dilution rate
    r = fill(rand(), Nr)

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
    K = rand(K_dist, Nr)

    # leakage now. Lets assume its a pretty flat probability distribution
    leak = Beta(0.2 / l_sparsity, 0.2)
    l = rand(leak, (Ns, Nr))

    # most values around 0. This is essentially a proxy for sparsity
    c_i_alpha = Beta(0.5 / c_sparsity, 0.5)
    c = rand(c_i_alpha, (Ns, Nr))

    # finally, the most complicated distribution
    D = fill(0.0, (Ns, Nr, Nr))
    for i in 1:Ns
        for j in 1:Nr
            if c[i, j] > 0
                flag = true
                while flag
                    for k in 1:Nr
                        if j == k
                            D[i, k, j] = 0.0
                        else
                            D[i, k, j] = rand(Beta(0.5 / (Nr / 5), 0.5))
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

    Ds = fill(0.0, (Ns + Nr))
    Ds[1:Ns] .= 1e-5
    Ds[1+Ns] = 100
    Ds[Ns+2:Ns+Nr] .= 10

    return r, m, K, l, c, D, Ds
end
export rg_stevens1_original

struct RSGStevens1
    Ns::Int
    Nr::Int
    c_sparsity::Float64
    l_sparsity::Float64
    usenthreads::Union{Nothing,Int}
    function RSGStevens1(Ns, Nr, c_sparsity, l_sparsity, usenthreads=nothing)
        new(Ns, Nr, c_sparsity, l_sparsity, usenthreads)
    end
end
function (rsg::RSGStevens1)()
    # as usual
    g = fill(1.0, rsg.Ns)
    w = fill(1.0, rsg.Nr)

    # universal death rate
    m = fill(rand(), rsg.Ns)

    # for simplicity, lets start with a single fed resource
    # chemostat feed rate 
    #K = fill(0.,M)
    #K[1] = 1.

    # lets allow resources some variability
    #K_dist = truncated(Normal(0.5,0.1), 0.0, 1.0)
    K_dist = Beta(0.1, 0.3)
    K = rand(K_dist, rsg.Nr)

    # constant dilution rate
    r = fill(rand(), rsg.Nr)

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
