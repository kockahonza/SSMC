module RandomParamGenerators

using Reexport
@reexport using ..ModifiedMiCRM

using StatsBase
using Distributions

struct FakeMMiCRMParams{Ns,Nr,F,A,B} # number of strains and resource types
    # these are usually all 1 from dimensional reduction
    g::SizedVector{Ns,F}
    w::SizedVector{Nr,F}

    # strain props
    m::SizedVector{Ns,F}

    # resource props
    K::SizedVector{Nr,F}
    r::SizedVector{Nr,F}

    # complex, matrix params
    l::SizedMatrix{Ns,Nr,F,A}
    c::SizedMatrix{Ns,Nr,F,A}
    D::SizedArray{Tuple{Ns,Nr,Nr},F,3,B} # D[1,a,b] corresponds to b -> a
end
export FakeMMiCRMParams
struct SuperFakeMMiCRMParams{F} # number of strains and resource types
    # these are usually all 1 from dimensional reduction
    g::Vector{F}
    w::Vector{F}

    # strain props
    m::Vector{F}

    # resource props
    K::Vector{F}
    r::Vector{F}

    # complex, matrix params
    l::Matrix{F}
    c::Matrix{F}
    D::Array{F,3} # D[1,a,b] corresponds to b -> a
end
export SuperFakeMMiCRMParams

abstract type MMiCRMParamGenerator end
function generate_random_mmicrmparams(generator::MMiCRMParamGenerator)
    throw(ErrorException(@sprintf "no function generate_random_mmicrmparams defined for generator type %s" string(typeof(generator))))
end
export MMiCRMParamGenerator, generate_random_mmicrmparams

function test_generate(Ns, Nr)
    g = fill(1.0, Ns)
    w = fill(1.0, Nr)
    m = fill(1.0, Ns)
    K = fill(1.0, Nr)
    r = fill(1.0, Nr)

    l = fill(0.0, Ns, Nr)
    c = fill(0.0, Ns, Nr)
    D = fill(0.0, Ns, Nr, Nr)

    SuperFakeMMiCRMParams(g, w, m, K, r, l, c, D)
end
export test_generate

function test_generate2(Ns, Nr)
    g = fill(1.0, Ns)
    w = fill(1.0, Nr)
    m = fill(1.0, Ns)
    K = fill(1.0, Nr)
    r = fill(1.0, Nr)

    l = fill(0.0, Ns, Nr)
    c = fill(0.0, Ns, Nr)
    D = fill(0.0, Ns, Nr, Nr)

    SuperFakeMMiCRMParams(g, w, m, K, r, l, c, D)
end
export test_generate2

function naive_test_generate(Ns, Nr;
    num_resources_per_species=4
)
    g = fill(1.0, Ns)
    w = fill(1.0, Nr)
    m = fill(1.0, Ns)
    K = fill(1.0, Nr)
    r = fill(1.0, Nr)

    l = fill(0.0, Ns, Nr)
    c = fill(0.0, Ns, Nr)
    D = fill(0.0, Ns, Nr, Nr)

    SuperFakeMMiCRMParams(g, w, m, K, r, l, c, D)
end
export naive_test_generate

end
