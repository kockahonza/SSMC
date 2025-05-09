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

struct SRGJans1{Dm,Dr,DDN,DDR,DK,Ds1,Ds2,Dc,Dl}
    Ns::Int
    Nr::Int

    m::Dm
    r::Dr
    DS::DDN
    DR::DDR

    Kp::Float64
    K::DK

    num_used_resources::Ds1
    num_byproducts::Ds2

    c::Dc
    l::Dl

    usenthreads::Union{Nothing,Int}
    function SRGJans1(Ns, Nr;
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
function (srg::SRGJans1)()
    # as usual
    g = fill(1.0, srg.Ns)
    w = fill(1.0, srg.Nr)

    m = rand(srg.m, srg.Ns)
    r = rand(srg.r, srg.Nr)

    K = fill(0.0, srg.Nr)
    for a in 1:srg.Nr
        if rand() < srg.Kp
            K[a] = rand(srg.K)
        end
    end

    l = fill(0.0, (srg.Ns, srg.Nr))
    c = fill(0.0, (srg.Ns, srg.Nr))
    D = fill(0.0, (srg.Ns, srg.Nr, srg.Nr))
    for i in 1:srg.Ns
        num_resources = rand(srg.num_used_resources)
        consumed_resources = sample(1:srg.Nr, num_resources; replace=false)
        for cr in consumed_resources
            c[i, cr] = abs(rand(srg.c))
            l[i, cr] = rand(srg.l)

            num_byproducts = rand(srg.num_byproducts)
            byproducts = sample(1:srg.Nr, num_byproducts; replace=false)
            bp_Ds = rand(length(byproducts))
            bp_Ds ./= sum(bp_Ds)
            for (bp, bp_D) in zip(byproducts, bp_Ds)
                D[i, bp, cr] = bp_D
            end
        end
    end

    DS = rand(srg.DS, srg.Ns)
    DR = rand(srg.DR, srg.Nr)
    # DR = fill(0.0, srg.Nr)
    # for a in 1:srg.Nr
    #     if K[a] != 0.0
    #         DR[a] = rand(srg.DR)
    #     end
    # end
    Ds = vcat(DS, DR)

    mmicrm_params = BMMiCRMParams(g, w, m, K, r, l, c, D, srg.usenthreads)
    BSMMiCRMParams(mmicrm_params, Ds)
end
export SRGJans1

struct SRGJans2{Dm,Dr,DDN,DDR,DK,Dci,Dli,Dcb,Dlb}
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
    function SRGJans2(Ns, Nr;
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
function (srg::SRGJans2)()
    # as usual
    g = fill(1.0, srg.Ns)
    w = fill(1.0, srg.Nr)

    m = rand(srg.m, srg.Ns)
    r = rand(srg.r, srg.Nr)

    K = fill(0.0, srg.Nr)
    for a in 1:srg.Nr
        if rand() < srg.Kp
            K[a] = rand(srg.K)
        end
    end

    l = fill(0.0, (srg.Ns, srg.Nr))
    c = fill(0.0, (srg.Ns, srg.Nr))
    D = fill(0.0, (srg.Ns, srg.Nr, srg.Nr))
    for i in 1:srg.Ns
        num_resources = rand(srg.num_used_resources)
        consumed_resources = sample(1:srg.Nr, num_resources; replace=false)
        for cr in consumed_resources
            c[i, cr] = abs(rand(srg.c))
            l[i, cr] = rand(srg.l)

            num_byproducts = rand(srg.num_byproducts)
            byproducts = sample(1:srg.Nr, num_byproducts; replace=false)
            bp_Ds = rand(length(byproducts))
            bp_Ds ./= sum(bp_Ds)
            for (bp, bp_D) in zip(byproducts, bp_Ds)
                D[i, bp, cr] = bp_D
            end
        end
    end

    DS = rand(srg.DS, srg.Ns)
    DR = rand(srg.DR, srg.Nr)
    # DR = fill(0.0, srg.Nr)
    # for a in 1:srg.Nr
    #     if K[a] != 0.0
    #         DR[a] = rand(srg.DR)
    #     end
    # end
    Ds = vcat(DS, DR)

    mmicrm_params = BMMiCRMParams(g, w, m, K, r, l, c, D, srg.usenthreads)
    BSMMiCRMParams(mmicrm_params, Ds)
end
export SRGJans2

end
