"""Smart function for making an initial state"""
function make_u0_smart(params::AbstractMMiCRMParams;
    u0=:maxNs, u0rand=nothing
)
    Ns, Nr = get_Ns(params)
    if isa(u0, Number) || isa(u0, AbstractArray)
        u0 = smart_val(u0, make_u0_steadyR(params), Ns + Nr)
    else
        u0name, u0args = split_name_args(u0)

        if u0name == :uniE
            u0 = make_u0_uniE(params)
        elseif u0name == :steadyR
            u0 = make_u0_steadyR(params)
        elseif u0name == :onlyN
            u0 = make_u0_onlyN(params)
        elseif u0name == :maxNs
            u0 = make_u0_maxNs(params)
        else
            throw(ArgumentError(@sprintf "cannot correctly parse u0name %s" string(u0name)))
        end
    end

    if !isnothing(u0rand)
        for i in eachindex(u0)
            u0[i] *= 1 + u0rand * (2 * rand() - 1)
        end
    end

    u0
end
make_u0_uniE(p::AbstractMMiCRMParams) = Vector(vcat(p.g, 1.0 ./ p.w))
make_u0_steadyR(p::AbstractMMiCRMParams) = Vector(vcat(p.g, p.K ./ p.r))
function make_u0_onlyN(p::AbstractMMiCRMParams)
    _, Nr = get_Ns(p)
    vcat(p.g, fill(0.0, Nr))
end
function make_u0_maxNs(p::AbstractMMiCRMParams)
    _, Nr = get_Ns(p)
    total_E_income = sum(p.K .* p.w)
    u_N = total_E_income ./ (p.g .* p.m)
    vcat(u_N, fill(0.0, Nr))
end
export make_u0_smart

# Spatial u0
function expand_u0_to_size(size::Tuple, nospaceu0)
    u0 = Array{eltype(nospaceu0),1 + length(size)}(undef, length(nospaceu0), size...)
    for ci in CartesianIndices(size)
        u0[:, ci] .= nospaceu0
    end
    u0
end
export expand_u0_to_size

# Perturbations
function perturb_u0_uniform(Ns, Nr, u0, e_s=nothing, e_r=nothing)
    N = Ns + Nr

    epsilon = if isnothing(e_r)
        smart_val(e_s, nothing, N)
    else
        vcat(smart_val(e_s, nothing, Ns), smart_val(e_r, nothing, Nr))
    end

    rand_pm_one = (2 .* rand(size(u0)...) .- 1)

    u0 .+ rand_pm_one .* epsilon
end
function perturb_u0_uniform_prop(Ns, Nr, u0, e_s=nothing, e_r=nothing)
    N = Ns + Nr

    epsilon = if isnothing(e_r)
        smart_val(e_s, nothing, N)
    else
        vcat(smart_val(e_s, nothing, Ns), smart_val(e_r, nothing, Nr))
    end

    rand_pm_one = (2 .* rand(size(u0)...) .- 1)
    # rand_pm_one = randn(size(u0)...)

    u0 .* (1.0 .+ rand_pm_one .* epsilon)
end
export perturb_u0_uniform, perturb_u0_uniform_prop
