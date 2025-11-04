using DimensionalData

function mapdim(func, dimarray, dimname, newdimname=dimname)
    set(dimarray, dimname => Dim{newdimname}(func.(dims(dimarray, dimname))))
end
