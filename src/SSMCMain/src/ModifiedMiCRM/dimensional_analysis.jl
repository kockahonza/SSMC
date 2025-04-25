function da_minmax_timescales_params_only(p::MMiCRMParams; include_infs=true)
    Ns, Nr = get_Ns(p)

    min = +Inf
    max = -Inf

    if include_infs
        test_min = x -> x < min
        test_max = x -> x > max
    else
        test_min = x -> -Inf < x < min
        test_max = x -> max < x < Inf
    end

    for i in 1:Ns
        v = 1 / p.m[i]
        if test_min(v)
            min = v
        end
        if test_max(v)
            max = v
        end
    end
    for a in 1:Nr
        v = 1 / p.r[a]
        if test_min(v)
            min = v
        end
        if test_max(v)
            max = v
        end
    end
    for i in 1:Ns
        for a in 1:Nr
            for b in 1:Nr
                v = 1 / sqrt(p.K[a] * p.c[i, b])
                if test_min(v)
                    min = v
                end
                if test_max(v)
                    max = v
                end
            end
        end
    end

    min, max
end
export da_minmax_timescales_params_only

function da_minmax_timescales_ss_based(p::MMiCRMParams, ss; include_infs=true)
    Ns, Nr = get_Ns(p)

    min = +Inf
    max = -Inf

    if include_infs
        test_min = x -> x < min
        test_max = x -> x > max
    else
        test_min = x -> -Inf < x < min
        test_max = x -> max < x < Inf
    end

    for b in 1:Nr
        for i in 1:Ns
            v = ss[i] / p.K[b]
            if test_min(v)
                min = v
            end
            if test_max(v)
                max = v
            end
        end
        for a in 1:Nr
            v = ss[Ns+a] / p.K[b]
            if test_min(v)
                min = v
            end
            if test_max(v)
                max = v
            end
        end
    end

    for j in 1:Ns
        for b in 1:Nr
            for i in 1:Ns
                v = 1 / (p.c[j, b] * ss[i])
                if test_min(v)
                    min = v
                end
                if test_max(v)
                    max = v
                end
            end
            for a in 1:Nr
                v = 1 / (p.c[j, b] * ss[Ns+a])
                if test_min(v)
                    min = v
                end
                if test_max(v)
                    max = v
                end
            end
        end
    end

    min, max
end
export da_minmax_timescales_ss_based

function da_minmax_timescales_simple(p::MMiCRMParams, ss; kwargs...)
    a, b = da_minmax_timescales_params_only(p; kwargs...)
    c, d = da_minmax_timescales_ss_based(p, ss; kwargs...)
    min(a, c), max(b, d)
end
export da_minmax_timescales_simple

function da_get_diff_lengthscales(Ds, timescales; include_infs=true)
    min = +Inf
    max = -Inf

    if include_infs
        test_min = x -> x < min
        test_max = x -> x > max
    else
        test_min = x -> -Inf < x < min
        test_max = x -> max < x < Inf
    end

    for D in Ds
        for t in timescales
            v = sqrt(D * t)
            if test_min(v)
                min = v
            end
            if test_max(v)
                max = v
            end
        end
    end

    min, max
end
export da_get_diff_lengthscales

function da_get_diff_lengthscales_simple(p::MMiCRMParams, Ds, ss; kwargs...)
    ts = da_minmax_timescales_simple(p, ss; kwargs...)
    da_get_diff_lengthscales(Ds, ts; kwargs...)
end
export da_get_diff_lengthscales_simple
