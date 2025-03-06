module SSMC

using REPL.TerminalMenus

using StaticArrays

################################################################################
# Utility bits
################################################################################
function uninplace(f)
    function (u, args...)
        du = similar(u)
        f(du, u, args...)
        du
    end
end
export uninplace

function smart_val(passed_val, default, shape...)
    val = isnothing(passed_val) ? default : passed_val
    if isnothing(val)
        throw(ArgumentError("encountered a nothing argument which does not have a default"))
    end
    if !isa(val, AbstractArray)
        val = fill(val, shape)
    end
    val
end
smart_sval(pv, d, shape...) = SArray{Tuple{shape...}}(smart_val(pv, d, shape...))
split_name_args(x) = x, ()
split_name_args(t::Tuple) = t[1], t[2:end]
export smart_val, smart_sval, split_name_args

function setup_ranges(ranges...; func=identity)
    cis = CartesianIndices(length.(ranges))
    function (i)
        func([r[i] for (r, i) in zip(ranges, cis[i].I)])
    end, cis
end
export setup_ranges

function wait_till_confirm()
    menu = RadioMenu(["continue", "maybe exit"], pagesize=2)
    request(menu) == 1
end
export wait_till_confirm

################################################################################
# Other modules/files
################################################################################
include("MLSolver.jl")
include("BasicMiCRM.jl")

end # module SSMC
