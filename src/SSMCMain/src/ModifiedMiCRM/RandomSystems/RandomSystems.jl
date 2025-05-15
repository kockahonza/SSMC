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
include("stevens_first.jl")

include("jans_first.jl")

include("stevens_marsland.jl")

end
