module SpaceMMiCRM

using Reexport
@reexport using ..SSMCMain
@reexport using ..SSMCMain.ModifiedMiCRM

# """Spatial Modified MiCRM model params"""
# struct SMMiCRMParams{D,Ns,Nr,F,S} # D is dimensionality, S=Ns+Nr
#     mmicrm_params::MMiCRMParams{Ns,Nr,F}
#     diff::SVector{S,F}
#     function SMMiCRMParams{D,Ns,Nr,F}(
#         mmicrm_params::MMiCRMParams{Ns,Nr,F},
#         diff::SVector{S,F}
#     )
#         if S == (Ns + Nr)
#             new{D,Ns,Nr,F,S}(mmicrm_params, diff)
#         else
#             throw(ArgumentError("passed mmicrm_params and diff are not compatible"))
#         end
#     end
# end
# export SMMiCRMParams



end
