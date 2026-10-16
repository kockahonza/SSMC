function print_spatial_solution_stats(sol)
    println(sol.retcode)
    @printf "nf is %d\n" sol.destats.nf
    @printf "solution has %d saved states\n" length(sol.u)
    max_resid = maximum(abs, uninplace(sol.prob.f)(sol.u[end], sol.prob.p))
    @printf "max resid is %g\n" max_resid

    mm = minimum(minimum, sol.u)
    if mm < 0
        @warn (@sprintf "reaching negative values, minimum is %g" mm)
    end
end
export print_spatial_solution_stats

function get_space_axes(space_size, dx=1.0; collect=true)
    dx = smart_val(dx, nothing, length(space_size))
    xx = tuple(((dx/2):dx:((s*dx-dx/2)) for (s, dx) in zip(space_size, dx))...)
    if collect
        map(Base.collect, xx)
    else
        xx
    end
end
get_space_axes(sample_array, args...; kwargs...) = get_space_axes(size(sample_array), args...; kwargs...)
export get_space_axes

function get_u_axes(u, args...; kwargs...)
    get_space_axes(size(u)[2:end], args...; kwargs...)
end
export get_u_axes

function resample_cartesian_u(u, space::CartesianSpace, Ns...)
    if length(Ns) == 1
        Ns = Ns[1]
    else
        Ns = collect(Ns)
    end
    Ns = smart_val(Ns, nothing, ndims(u) - 1)

    ranges = get_u_axes(u, space.dx)
    new_ranges = [LinRange(or[1], or[end], N) for (or, N) in zip(ranges, Ns)]
    new_dx = [r[2] - r[1] for r in new_ranges]
    new_space = change_cartesianspace_dx(space, new_dx)

    ru = similar(u, size(u)[1], Ns...)

    for i in axes(u, 1)
        old_ui = selectdim(u, 1, i)
        ii = linear_interpolation(ranges, old_ui)
        selectdim(ru, 1, i) .= ii(new_ranges...)
    end

    ru, new_space
end
export resample_cartesian_u

function clamp_ss!(ss, val=eps())
    for i in eachindex(ss)
        x = ss[i]
        if x < 0.0
            ss[i] = val
            if x < -100 * eps()
                @warn (@sprintf "clamping %g to 0 which is more than %g" x 100 * eps())
            end
        end
    end
    ss
end
function clamp_ss(ss, args...)
    ss = copy(ss)
    clamp_ss!(ss, args...)
end
export clamp_ss!, clamp_ss

function base10_lognormal(b10m, b10std)
    LogNormal(b10m * log(10), b10std * log(10))
end
export base10_lognormal

function num_survivors_in_space(u, Ns, threshold=1e5 * eps(eltype(u)))
    num = 0
    for i in 1:Ns
        us = @view u[i, :]
        for v in us
            if v > threshold
                num += 1
                break
            end
        end
    end
    num
end
export num_survivors_in_space

"""
Drops strains below `threshold` in `final_state` and resources no survivor takes
up or produces. Returns `(; params, final_state, keep_s, keep_r)`.
"""
function reduce_extinct(ps::BMMiCRMParams, final_state, threshold)
    Ns, Nr = get_Ns(ps)
    if length(final_state) != Ns + Nr
        throw(ArgumentError(@sprintf(
            "final_state has length %d, expected %d for a %d strain, %d resource system",
            length(final_state), Ns + Nr, Ns, Nr)))
    end

    keep_s = [i for i in 1:Ns if final_state[i] >= threshold]
    uses(i, a) = !iszero(ps.c[i, a]) || any(1:Nr) do b
        !iszero(ps.D[i, a, b]) && !iszero(ps.l[i, b]) && !iszero(ps.c[i, b])
    end
    keep_r = [a for a in 1:Nr if any(i -> uses(i, a), keep_s)]

    rps = BMMiCRMParams(
        ps.g[keep_s], ps.w[keep_r], ps.m[keep_s],
        ps.K[keep_r], ps.r[keep_r],
        ps.l[keep_s, keep_r], ps.c[keep_s, keep_r], ps.D[keep_s, keep_r, keep_r],
        ps.usenthreads,
    )

    (; params=rps, final_state=[final_state[keep_s]; final_state[Ns .+ keep_r]], keep_s, keep_r)
end
export reduce_extinct

"""
Terminates with `ReturnCode.Stalled` if the min `max|du|` hasn't dropped 10x over
a decade of `t` and is still above `resid_floor`. Make a fresh one per run.
"""
function make_stagnation_callback(ps, resid_floor; check_every=1000, t_start=10.)
    du = zeros(sum(get_Ns(ps)))
    n, tnext, rmin, rprev = 0, t_start, Inf, Inf
    DiscreteCallback(
        function (u, t, _)
            ((n += 1) % check_every == 0 && t > t_start) || return false
            mmicrmfunc!(du, u, ps)
            rmin = min(rmin, maximum(abs, du))
            t < tnext && return false
            stalled = (rmin > 0.1 * rprev) && (rmin > resid_floor)
            rprev, rmin, tnext = rmin, Inf, 10 * tnext
            stalled
        end,
        i -> terminate!(i, ReturnCode.Stalled);
        save_positions=(false, false),
    )
end
export make_stagnation_callback

"""
Peaks of a 1d periodic field using Peaks.jl. The field is rotated so its global
min is at the array edge, which makes prominences and widths exact, and padded
with `w` wrapped points each side so findmaxima sees every point's full window.
`edges` are fractional gridpoint indices with `left` in [1, sN+1), `right` can
go past sN if it wraps.
"""
function find_peaks_periodic(N;
    w=1, min_prom=nothing, min_width=nothing, relheight=0.5, min_height=nothing
)
    sN = length(N)
    nopeaks = (; indices=Int[], heights=Float64[], proms=Float64[], widths=Float64[],
        edges=Tuple{Float64,Float64}[])

    s = argmin(N) - 1
    Np = N[mod1.(((1-w):(sN+w)) .+ s, sN)] # padded index p is original index p - w + s

    # strict=false so plateaus get a prominence and width rather than 0 and NaN
    pks = findmaxima(Np, w) |> peakproms!(; min=min_prom, strict=false)
    isempty(pks.indices) && return nopeaks # peakwidths! errors on no peaks
    pks = pks |> peakwidths!(; min=min_width, relheight, strict=false) |> peakheights!(; min=min_height)

    # drop the copies found in the padding
    keep = findall(p -> w < p <= sN + w, pks.indices)
    edges = map(pks.edges[keep]) do (l, r)
        k = fld(l - w + s - 1, sN)
        (l - w + s - k * sN, r - w + s - k * sN)
    end
    (; indices=mod1.(pks.indices[keep] .- w .+ s, sN), heights=pks.heights[keep],
        proms=pks.proms[keep], widths=pks.widths[keep], edges)
end
export find_peaks_periodic
