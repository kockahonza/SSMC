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
