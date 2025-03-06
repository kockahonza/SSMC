"""Symbolic solver using MathLink"""
module MLSolver
using ..SSMC

using Printf

using DifferentialEquations
using Symbolics, SymbolicsMathLink

function symbolic_solve_ode_ml(p::ODEProblem; raw=true)
    n = length(p.u0)
    su = [Symbolics.variable(Symbol(@sprintf "u%d" i)) for i in 1:n]
    eqs = if isinplace(p)
        uninplace(p.f)(su, p.p, 0)
    else
        p.f(su, p.p, 0)
    end
    ws = wcall("Solve", eqs .~ 0, su)
    if raw
        ws
    else
        map.(x -> Symbolics.unwrap.(x[2]), ws)
    end
end
export symbolic_solve_ode_ml

# Alternative more fancy way of doing things...
# import DifferentialEquations: solve
#
# struct MLSolverAlg end
# export MLSolverAlg
#
# function solve(p::ODEProblem, _::MLSolverAlg)
#     n = length(p.u0)
#     su = [Symbolics.variable(Symbol(@sprintf "u%d" i)) for i in 1:n]
#     eqs = if isinplace(p)
#         uninplace(p.f)(su, p.p, 0)
#     else
#         p.f(su, p.p, 0)
#     end
#     ws = wcall("Solve", eqs .~ 0, su)
#     map.(x -> Symbolics.unwrap.(x[2]), ws)
# end

end


