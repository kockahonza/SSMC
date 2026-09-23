function instability_stats(cm;
    unstable_codes=[2],
    other_good_codes=[101, 1],
    return_confint_only=true,
    warn_bad_data_ratio=0.9
)
    unstable_runs = sum(c -> get(cm, c, 0), unstable_codes)
    other_good_runs = sum(c -> get(cm, c, 0), other_good_codes)

    good_runs = unstable_runs + other_good_runs
    runs = sum(values(cm))
    if (good_runs / runs) < warn_bad_data_ratio
        @warn "Good data ratio is low: $(good_runs / runs) < $warn_bad_data_ratio"
    end

    bt = BinomialTest(unstable_runs, unstable_runs + other_good_runs)

    if return_confint_only
        (bt.x / bt.n), confint(bt; method=:wilson)
    else
        bt
    end
end
function instability_stats(rslt_codes::Vector; kwargs...)
    instability_stats(countmap(rslt_codes); kwargs...)
end
export instability_stats

function rsg_stats1(rsg)
    ps = rsg()

    f = Figure()
    ax1 = Axis(f[1, 1]; xlabel="c")
    hist!(ax1, ps.c[:])
    ax2 = Axis(f[1, 2]; xlabel="l")
    hist!(ax2, ps.l[:])
    ax3 = Axis(f[2, 1]; xlabel="D")
    hist!(ax3, ps.D[:])
    ax4 = Axis(f[2, 2]; xlabel="m")
    hist!(ax4, ps.m[:])

    f
end
export rsg_stats1

"""
Uniform in logit space between lo and hi, for sampling l values near 1.
Only quantile and pdf are defined, rand falls back on quantile.
"""
struct LogitUniform <: ContinuousUnivariateDistribution
    lo::Float64
    hi::Float64
end
_logit(p) = log(p / (1 - p))
function Distributions.quantile(d::LogitUniform, q::Real)
    llo, lhi = _logit(d.lo), _logit(d.hi)
    1 / (1 + exp(-(llo + q * (lhi - llo))))
end
function Distributions.pdf(d::LogitUniform, x::Real)
    if d.lo <= x <= d.hi
        1 / ((_logit(d.hi) - _logit(d.lo)) * x * (1 - x))
    else
        0.0
    end
end
export LogitUniform
