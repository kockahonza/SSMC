module SSMCMain

using Reexport
@reexport using Printf
@reexport using StaticArrays
@reexport using DifferentialEquations, NonlinearSolve, Optimization
@reexport using LinearAlgebra
@reexport using Makie, LaTeXStrings

using REPL.TerminalMenus
using Logging

using PrettyTables

import Base: show

projectdir(args...) = abspath(joinpath(pkgdir(SSMCMain), "../../", args...))
scriptsdir(args...) = joinpath(projectdir(), "scripts", args...)
datadir(args...) = joinpath(projectdir(), "data", args...)
export projectdir, scriptsdir, datadir

################################################################################
# Other modules/files
################################################################################
include("util.jl")

# Main stuff!
include("ModifiedMiCRM/ModifiedMiCRM.jl")
include("ModifiedMiCRM/SpaceMMiCRM.jl")

# Old stuff, just keeping for old times
include("archive/BasicMiCRM.jl")

end # module SSMCMain
