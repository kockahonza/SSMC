module RandomSystems

using Reexport
@reexport using ..ModifiedMiCRM

using StatsBase
using Distributions

abstract type MMiCRMParamGenerator end
function generate_random_mmicrmparams(generator::MMiCRMParamGenerator)
    throw(ErrorException(@sprintf "no function generate_random_mmicrmparams defined for generator type %s" string(typeof(generator))))
end
export MMiCRMParamGenerator, generate_random_mmicrmparams

################################################################################
# Methods that will sample generators
################################################################################
function rg_sample_spatial(rg, num_repeats, ss_finder, test_func; save_succ=false)
    num_succ = 0
    if save_succ
        succ = []
    end

    for i in 1:num_repeats
        smmicrm_params = rg()
        ss = ss_finder(smmicrm_params)
        if test_func(smmicrm_params, ss)
            num_succ += 1
            if save_succ
                push!(succ, smmicrm_params)
            end
        end
    end

    if save_succ
        num_succ, succ
    else
        num_succ
    end
end
export rg_sample_spatial

################################################################################
# Sampling generators
################################################################################
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

struct SRGStevens1
    Ns::Int
    Nr::Int
    c_sparsity::Float64
    l_sparsity::Float64
    usenthreads::Union{Nothing,Int}
    function SRGStevens1(Ns, Nr, c_sparsity, l_sparsity, usenthreads=nothing)
        new(Ns, Nr, c_sparsity, l_sparsity, usenthreads)
    end
end
function (srg::SRGStevens1)()
    # as usual
    g = fill(1.0, srg.Ns)
    w = fill(1.0, srg.Nr)

    # universal death rate
    m = fill(rand(), srg.Ns)

    # for simplicity, lets start with a single fed resource
    # chemostat feed rate 
    #K = fill(0.,M)
    #K[1] = 1.

    # lets allow resources some variability
    #K_dist = truncated(Normal(0.5,0.1), 0.0, 1.0)
    K_dist = Beta(0.1, 0.3)
    K = rand(K_dist, srg.Nr)

    # constant dilution rate
    r = fill(rand(), srg.Nr)

    # leakage now. Lets assume its a pretty flat probability distribution
    leak = Beta(0.2 / srg.l_sparsity, 0.2)
    l = rand(leak, (srg.Ns, srg.Nr))

    # most values around 0. This is essentially a proxy for sparsity
    c_i_alpha = Beta(0.5 / srg.c_sparsity, 0.5)
    c = rand(c_i_alpha, (srg.Ns, srg.Nr))

    # finally, the most complicated distribution
    D = fill(0.0, (srg.Ns, srg.Nr, srg.Nr))
    for i in 1:srg.Ns
        for j in 1:srg.Nr
            if c[i, j] > 0
                flag = true
                while flag
                    for k in 1:srg.Nr
                        if j == k
                            D[i, k, j] = 0.0
                        else
                            D[i, k, j] = rand(Beta(0.5 / (srg.Nr / 5), 0.5))
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

    Ds = fill(0.0, (srg.Ns + srg.Nr))
    Ds[1:srg.Ns] .= 1e-5
    Ds[1+srg.Ns] = 100
    Ds[srg.Ns+2:srg.Ns+srg.Nr] .= 10

    # return a spatial mmicrm params struct without a concrete space and with threading on the nospace level
    p = BMMiCRMParams(g, w, m, K, r, l, c, D, srg.usenthreads)
    BSMMiCRMParams(p, Ds)
end
export SRGStevens1

end
