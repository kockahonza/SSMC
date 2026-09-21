"""
    TernaryColormaps

Ternary (three-component compositional) colour systems for Makie, in the spirit of
the R package `tricolore` and `pyrolite`'s `ternary_color`.

A `TernaryColormap` holds three corner colours and blends them barycentrically in a
chosen colour space. Reading tricolore's source, its "ternary balance scheme" is
exactly this: a weighted mix in CIE Lab of three corners placed at equal lightness
and chroma and 120-degree-spaced hues. pyrolite's version is the same mix taken in
sRGB. So one pipeline covers both, and mixing in Oklab (the default here) gives the
most even hue transitions and a properly neutral centre.

Makie colormaps are scalar -> colour, so a ternary map cannot be passed as a
`colormap`. A `TernaryColormap` is instead a callable you broadcast to produce a
colour per data point:

    cm = TernaryColormap()
    scatter!(ax, x, y; color = cm.(p1, p2, p3))

`ternary_legend` / `ternary_legend!` draw the colorbar-analogue: the filled triangle
with configurable frame, grid, ticks, tick labels, corner labels and axis arrows.

This file is a self-contained library; `include` it and `using .TernaryColormaps`.
"""
module TernaryColormaps

using Colors
using Makie

export TernaryColormap, balanced_corners, bary2cart, ternary_legend, ternary_legend!

# --------------------------------------------------------------- colour spaces

"Colour spaces the barycentric mix can be taken in."
const MIXING_SPACES = (:oklab, :lab, :luv, :srgb, :linear_rgb)

# Each space needs a forward map colour -> 3 coordinates and its inverse. Mixing
# happens on the coordinates, so the space fully determines how blends look.
_fwd(::Val{:oklab}, c) = (x = Oklab(c); (Float64(x.l), Float64(x.a), Float64(x.b)))
_inv(::Val{:oklab}, v) = RGB{Float64}(Oklab(v[1], v[2], v[3]))

_fwd(::Val{:lab}, c) = (x = Lab(c); (Float64(x.l), Float64(x.a), Float64(x.b)))
_inv(::Val{:lab}, v) = RGB{Float64}(Lab(v[1], v[2], v[3]))

_fwd(::Val{:luv}, c) = (x = Luv(c); (Float64(x.l), Float64(x.u), Float64(x.v)))
_inv(::Val{:luv}, v) = RGB{Float64}(Luv(v[1], v[2], v[3]))

_fwd(::Val{:srgb}, c) = (x = RGB{Float64}(c); (x.r, x.g, x.b))
_inv(::Val{:srgb}, v) = RGB{Float64}(v[1], v[2], v[3])

_s2l(u) = u <= 0.04045 ? u / 12.92 : ((u + 0.055) / 1.055)^2.4
_l2s(u) = (u = max(u, 0.0); u <= 0.0031308 ? 12.92u : 1.055 * u^(1 / 2.4) - 0.055)
_fwd(::Val{:linear_rgb}, c) = (x = RGB{Float64}(c); (_s2l(x.r), _s2l(x.g), _s2l(x.b)))
_inv(::Val{:linear_rgb}, v) = RGB{Float64}(_l2s(v[1]), _l2s(v[2]), _l2s(v[3]))

_tocolor(c) = (x = Makie.to_color(c); RGB{Float64}(red(x), green(x), blue(x)))
_ingamut(c::RGB) = Colors.mapc(Colors.clamp01, c)

"""
    balanced_corners(; hue=30, chroma=0.115, lightness=0.72)

Three corner colours of equal lightness and chroma with hues 120 degrees apart in
Oklab, i.e. tricolore's balanced default. `hue` (degrees) rotates all three,
`chroma` sets how vivid the corners are and `lightness` how bright. Returns a
3-tuple of `RGB{Float64}` suitable as the first three arguments of
[`TernaryColormap`](@ref).
"""
function balanced_corners(; hue=30.0, chroma=0.115, lightness=0.72)
    ntuple(3) do i
        h = deg2rad(hue + 120 * (i - 1))
        _ingamut(RGB{Float64}(Oklab(lightness, chroma * cos(h), chroma * sin(h))))
    end
end

# ------------------------------------------------------------------ the colormap

"""
    TernaryColormap(c1, c2, c3; space=:oklab, power=1, center=nothing, saturation=1)
    TernaryColormap(; kwargs...)

A callable mapping a three-component composition to a colour by blending the corner
colours `c1`, `c2`, `c3` (anything Makie accepts as a colour). The no-corner form
uses [`balanced_corners`](@ref).

For a composition `p`, the pipeline is

1. normalise `p` to sum to 1 (so raw counts or fractions both work),
2. recentre on `center`: `p ./ center`, renormalised,
3. sharpen by `power`: `p .^ power`, renormalised,
4. mix the corner colours with those weights in `space`,
5. scale the result away from the neutral centre colour by `saturation`,
6. convert to sRGB and clamp into gamut.

Options
- `space`: one of `$(MIXING_SPACES)`. `:oklab` (default) and `:lab` are perceptual
  and give a neutral grey centre; `:lab` reproduces tricolore, `:srgb` reproduces
  pyrolite; `:linear_rgb` mixes like light.
- `power > 0`: stretches the blend. `> 1` pushes weight onto the dominant component,
  so colours saturate faster towards the corners; `< 1` flattens everything towards
  the centre. (tricolore calls this `spread`.)
- `center`: a composition to treat as the neutral centre, for data that clusters off
  the middle of the triangle. `nothing` means `(1/3, 1/3, 1/3)`, i.e. no shift.
- `saturation`: scales the distance from the centre colour in `space`. `0` collapses
  to a flat neutral, `1` leaves the mix alone, `> 1` exaggerates (then clamps).

Calling it

    cm(0.2, 0.3, 0.5)        # a single composition, as three numbers
    cm((0.2, 0.3, 0.5))      # ... or as a tuple / length-3 vector
    cm.(a, b, c)             # broadcast over three arrays -> Vector{RGB}
    cm(M)                    # an N x 3 matrix of rows -> Vector{RGB}

Note `cm.(p)` on a single length-3 vector broadcasts over its elements and is not
what you want; use `cm(p)`.
"""
struct TernaryColormap{S}
    corners::NTuple{3,RGB{Float64}}
    coords::NTuple{3,NTuple{3,Float64}}  # corners expressed in the mixing space
    neutral::NTuple{3,Float64}           # equal-weight mix, in the mixing space
    power::Float64
    center::NTuple{3,Float64}
    saturation::Float64
end

function TernaryColormap(c1, c2, c3;
    space=:oklab, power=1.0, center=nothing, saturation=1.0
)
    space in MIXING_SPACES ||
        throw(ArgumentError("space must be one of $(MIXING_SPACES), got :$space"))
    power > 0 || throw(ArgumentError("power must be positive, got $power"))

    ctr = if isnothing(center)
        (1 / 3, 1 / 3, 1 / 3)
    else
        length(center) == 3 || throw(ArgumentError("center must have 3 components"))
        all(>(0), center) ||
            throw(ArgumentError("center components must be strictly positive, got $center"))
        s = sum(center)
        (center[1] / s, center[2] / s, center[3] / s)
    end

    corners = (_tocolor(c1), _tocolor(c2), _tocolor(c3))
    coords = map(c -> _fwd(Val(space), c), corners)
    neutral = ntuple(k -> (coords[1][k] + coords[2][k] + coords[3][k]) / 3, 3)

    TernaryColormap{space}(corners, coords, neutral, Float64(power), ctr, Float64(saturation))
end

TernaryColormap(; kwargs...) = TernaryColormap(balanced_corners()...; kwargs...)

"The colour space a `TernaryColormap` blends in."
mixing_space(::TernaryColormap{S}) where {S} = S

function Base.show(io::IO, cm::TernaryColormap)
    print(io, "TernaryColormap(", join(("#" * hex(c) for c in cm.corners), ", "),
        "; space=:", mixing_space(cm), ", power=", cm.power,
        ", center=", cm.center, ", saturation=", cm.saturation, ")")
end

function _weights(cm::TernaryColormap, p::NTuple{3,Float64})
    any(x -> x < 0 || isnan(x), p) &&
        throw(ArgumentError("ternary components must be non-negative and finite, got $p"))
    s = p[1] + p[2] + p[3]
    s > 0 || throw(ArgumentError("ternary components must not all be zero"))
    q = (p[1] / s, p[2] / s, p[3] / s)

    if cm.center != (1 / 3, 1 / 3, 1 / 3)
        q = (q[1] / cm.center[1], q[2] / cm.center[2], q[3] / cm.center[3])
        s = q[1] + q[2] + q[3]
        q = (q[1] / s, q[2] / s, q[3] / s)
    end

    if cm.power != 1
        q = (q[1]^cm.power, q[2]^cm.power, q[3]^cm.power)
        s = q[1] + q[2] + q[3]
        q = (q[1] / s, q[2] / s, q[3] / s)
    end

    q
end

function (cm::TernaryColormap{S})(a::Real, b::Real, c::Real) where {S}
    w = _weights(cm, (Float64(a), Float64(b), Float64(c)))
    v = ntuple(k -> w[1] * cm.coords[1][k] + w[2] * cm.coords[2][k] + w[3] * cm.coords[3][k], 3)
    if cm.saturation != 1
        v = ntuple(k -> cm.neutral[k] + cm.saturation * (v[k] - cm.neutral[k]), 3)
    end
    _ingamut(_inv(Val(S), v))
end

function (cm::TernaryColormap)(p::Union{Tuple,AbstractVector})
    length(p) == 3 ||
        throw(ArgumentError("expected a 3-component composition, got length $(length(p))"))
    cm(p[1], p[2], p[3])
end

function (cm::TernaryColormap)(M::AbstractMatrix)
    size(M, 2) == 3 ||
        throw(ArgumentError("expected an N x 3 matrix of compositions, got $(size(M))"))
    [cm(M[i, 1], M[i, 2], M[i, 3]) for i in axes(M, 1)]
end

# -------------------------------------------------------------------- geometry

"Cartesian positions of the three corners: component 1 top, 2 bottom-left, 3 bottom-right."
const CORNERS = (Point2f(0.5, sqrt(3) / 2), Point2f(0.0, 0.0), Point2f(1.0, 0.0))
const CENTROID = Point2f(0.5, sqrt(3) / 6)

"""
    bary2cart(a, b, c) -> Point2f
    bary2cart(p) -> Point2f

Position of a composition in the unit triangle used by [`ternary_legend!`](@ref):
component 1 at the top corner, 2 bottom-left, 3 bottom-right. Inputs are normalised,
so counts work as well as fractions. Use this to draw your own data on a legend axis.
"""
function bary2cart(a::Real, b::Real, c::Real)
    s = a + b + c
    Point2f(
        (a * CORNERS[1][1] + b * CORNERS[2][1] + c * CORNERS[3][1]) / s,
        (a * CORNERS[1][2] + b * CORNERS[2][2] + c * CORNERS[3][2]) / s,
    )
end
bary2cart(p) = bary2cart(p[1], p[2], p[3])

# Outward unit normals of the edge each component's ticks live on, and the angle of
# that edge (used for :parallel tick label rotation).
const _TICK_NORMALS = (
    Point2f(-sqrt(3) / 2, 0.5),  # component 1 -> left edge
    Point2f(0.0, -1.0),          # component 2 -> bottom edge
    Point2f(sqrt(3) / 2, 0.5),   # component 3 -> right edge
)
const _EDGE_ANGLES = (pi / 3, 0.0, -pi / 3)

# Barycentric anchor of tick `t` on each component's edge.
_tick_bary(::Val{1}, t) = (t, 1 - t, 0.0)
_tick_bary(::Val{2}, t) = (0.0, t, 1 - t)
_tick_bary(::Val{3}, t) = (1 - t, 0.0, t)

# Endpoints of the gridline of constant value `t` for each component.
_grid_bary(::Val{1}, t) = ((t, 1 - t, 0.0), (t, 0.0, 1 - t))
_grid_bary(::Val{2}, t) = ((0.0, t, 1 - t), (1 - t, t, 0.0))
_grid_bary(::Val{3}, t) = ((1 - t, 0.0, t), (0.0, 1 - t, t))

# Direction of increase along each component's edge, as barycentric endpoints.
_arrow_bary(::Val{1}) = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
_arrow_bary(::Val{2}) = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
_arrow_bary(::Val{3}) = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))

# Triangular lattice of `n` subdivisions: vertex positions, their barycentric
# coordinates, and an nfaces x 3 index matrix.
function _tri_lattice(n::Integer)
    n >= 1 || throw(ArgumentError("resolution must be at least 1, got $n"))
    npts = (n + 1) * (n + 2) ÷ 2
    pts = Vector{Point2f}(undef, npts)
    bary = Vector{NTuple{3,Float64}}(undef, npts)
    idx = zeros(Int, n + 1, n + 1)

    k = 0
    for i in 0:n, j in 0:(n-i)
        k += 1
        idx[i+1, j+1] = k
        bary[k] = (i / n, j / n, (n - i - j) / n)
        pts[k] = bary2cart(bary[k])
    end

    faces = NTuple{3,Int}[]
    for i in 0:(n-1), j in 0:(n-i-1)
        push!(faces, (idx[i+1, j+1], idx[i+2, j+1], idx[i+1, j+2]))
        i + j <= n - 2 && push!(faces, (idx[i+2, j+1], idx[i+2, j+2], idx[i+1, j+2]))
    end

    pts, bary, _facematrix(faces)
end

function _facematrix(faces::Vector{NTuple{3,Int}})
    F = Matrix{Int}(undef, length(faces), 3)
    for (m, f) in enumerate(faces)
        F[m, 1], F[m, 2], F[m, 3] = f
    end
    F
end

# Same lattice but with each face given its own vertices, so each bin is a flat patch
# coloured by its centroid.
function _tri_bins(n::Integer, cm::TernaryColormap)
    n >= 1 || throw(ArgumentError("nbins must be at least 1, got $n"))
    pts = Point2f[]
    cols = RGB{Float64}[]
    faces = NTuple{3,Int}[]
    b(i, j) = (i / n, j / n, (n - i - j) / n)

    function emit(b1, b2, b3)
        base = length(pts)
        push!(pts, bary2cart(b1), bary2cart(b2), bary2cart(b3))
        col = cm(ntuple(k -> (b1[k] + b2[k] + b3[k]) / 3, 3))
        push!(cols, col, col, col)
        push!(faces, (base + 1, base + 2, base + 3))
    end

    for i in 0:(n-1), j in 0:(n-i-1)
        emit(b(i, j), b(i + 1, j), b(i, j + 1))
        i + j <= n - 2 && emit(b(i + 1, j), b(i + 1, j + 1), b(i, j + 1))
    end

    pts, cols, _facematrix(faces)
end

# Let `ticks`/`tickformat` be given either once for all three axes or per axis.
_percomponent(x) =
    if x isa Tuple{Any,Any,Any} && all(y -> y isa Union{AbstractVector,AbstractRange,Tuple}, x)
        x
    else
        (x, x, x)
    end

# --------------------------------------------------------------------- legend

"""
    ternary_legend!(target, cm::TernaryColormap; kwargs...) -> Axis

Draw `cm` as a filled triangle: the ternary equivalent of a colorbar. `target` is a
`GridPosition` (e.g. `fig[1, 2]`), a `Figure` (uses `fig[1, 1]`), or an existing
`Axis` to draw into (in which case aspect, limits and decorations are left alone,
and `padding`/`axiskwargs` do not apply). Returns the `Axis`, so you can keep
adding to it — use [`bary2cart`](@ref) to place your own compositions on it.

Fill
- `resolution = 128`: subdivisions of the smooth triangular mesh.
- `nbins = nothing`: if set, draw that many discrete bins per side instead of a
  smooth fill (tricolore's `breaks`).

Frame and grid
- `framevisible`, `framecolor`, `framewidth`
- `gridvisible`, `gridcolor`, `gridwidth`, `gridstyle`

Ticks. `ticks` and `tickformat` accept either one value for all three axes or a
3-tuple for per-axis control.
- `ticks = 0.2:0.2:0.8`, `tickformat`, `ticklabels = nothing` (explicit strings).
  The endpoints are left out by default because 0 and 1 land on corners, where two
  axes meet and their labels overlap; pass `0:0.2:1` if you want them.
- `ticksvisible`, `ticklength`, `tickwidth`, `tickcolor`
- `ticklabelsvisible`, `ticklabelsize`, `ticklabelcolor`, `ticklabeloffset`
- `ticklabelrotation`: `:horizontal`, `:parallel`, or an angle in radians

Corner labels
- `labels = ("1", "2", "3")`, `labelsvisible`, `labelsize`, `labelcolor`, `labeloffset`

Axis arrows, drawn alongside each edge in the direction the component increases
- `arrowsvisible = false`, `arrowcolor`, `arrowwidth`, `arrowsize`, `arrowoffset`,
  `arrowspan`, `arrowlabels = nothing` (defaults to `labels`), `arrowlabelsize`,
  `arrowlabeloffset`

Axis
- `padding = 0.15`: blank space around the triangle, a number or
  `(left, right, bottom, top)`.
- `axiskwargs = (;)`: passed through to the `Axis` constructor when one is created.
"""
function ternary_legend! end

function ternary_legend!(ax::Axis, cm::TernaryColormap;
    resolution=128,
    nbins=nothing,
    framevisible=true, framecolor=:black, framewidth=1.5,
    gridvisible=true, gridcolor=(:black, 0.3), gridwidth=0.75, gridstyle=:solid,
    ticks=0.2:0.2:0.8, tickformat=_default_tickformat, ticklabels=nothing,
    ticksvisible=true, ticklength=0.025, tickwidth=1.0, tickcolor=:black,
    ticklabelsvisible=true, ticklabelsize=11, ticklabelcolor=:black,
    ticklabeloffset=0.045, ticklabelrotation=:horizontal,
    labels=("1", "2", "3"), labelsvisible=true, labelsize=14, labelcolor=:black,
    labeloffset=0.10,
    arrowsvisible=false, arrowcolor=:black, arrowwidth=1.5, arrowsize=12,
    arrowoffset=0.16, arrowspan=(0.12, 0.88), arrowlabels=nothing,
    arrowlabelsize=12, arrowlabeloffset=0.05,
)
    _fill!(ax, cm, resolution, nbins)

    tickss = _percomponent(ticks)
    formats = _percomponent(tickformat)
    labelss = isnothing(ticklabels) ? nothing : _percomponent(ticklabels)

    gridvisible && _grid!(ax, tickss, gridcolor, gridwidth, gridstyle)
    _ticks!(ax, tickss, formats, labelss,
        ticksvisible, ticklength, tickwidth, tickcolor,
        ticklabelsvisible, ticklabelsize, ticklabelcolor, ticklabeloffset, ticklabelrotation)
    framevisible && _frame!(ax, framecolor, framewidth)
    labelsvisible && _corner_labels!(ax, labels, labelsize, labelcolor, labeloffset)
    arrowsvisible && _arrows!(ax, isnothing(arrowlabels) ? labels : arrowlabels,
        arrowcolor, arrowwidth, arrowsize, arrowoffset, arrowspan,
        arrowlabelsize, arrowlabeloffset)

    ax
end

function ternary_legend!(target, cm::TernaryColormap;
    padding=0.15, axiskwargs=(;), kwargs...
)
    gp = target isa Figure ? target[1, 1] : target
    ax = Axis(gp; aspect=DataAspect(), axiskwargs...)
    hidedecorations!(ax)
    hidespines!(ax)

    pl, pr, pb, pt = padding isa Real ? ntuple(_ -> Float64(padding), 4) : padding
    limits!(ax, -pl, 1 + pr, -pb, sqrt(3) / 2 + pt)

    ternary_legend!(ax, cm; kwargs...)
end

"""
    ternary_legend(cm::TernaryColormap; figure=(;), kwargs...) -> Figure

Convenience wrapper that makes a `Figure` and draws [`ternary_legend!`](@ref) into
it. `figure` is passed to the `Figure` constructor; everything else goes to
`ternary_legend!`. Use `ternary_legend!` when you need the `Axis` back.
"""
function ternary_legend(cm::TernaryColormap; figure=(;), kwargs...)
    fig = Figure(; figure...)
    ternary_legend!(fig[1, 1], cm; kwargs...)
    fig
end

_default_tickformat(t) = string(round(t; digits=3))

function _fill!(ax, cm, resolution, nbins)
    if isnothing(nbins)
        pts, bary, faces = _tri_lattice(resolution)
        mesh!(ax, pts, faces; color=[cm(b) for b in bary], shading=NoShading)
    else
        pts, cols, faces = _tri_bins(nbins, cm)
        mesh!(ax, pts, faces; color=cols, shading=NoShading)
    end
end

function _frame!(ax, color, width)
    lines!(ax, [CORNERS[1], CORNERS[2], CORNERS[3], CORNERS[1]]; color, linewidth=width)
end

function _grid!(ax, tickss, color, width, style)
    segs = Point2f[]
    for k in 1:3, t in tickss[k]
        0 < t < 1 || continue
        b1, b2 = _grid_bary(Val(k), t)
        push!(segs, bary2cart(b1), bary2cart(b2))
    end
    isempty(segs) || linesegments!(ax, segs; color, linewidth=width, linestyle=style)
end

function _ticks!(ax, tickss, formats, labelss,
    ticksvisible, ticklength, tickwidth, tickcolor,
    labelsvisible, labelsize, labelcolor, labeloffset, labelrotation
)
    segs = Point2f[]
    positions = Point2f[]
    strings = String[]
    rotations = Float64[]

    for k in 1:3
        n = _TICK_NORMALS[k]
        rot = if labelrotation === :horizontal
            0.0
        elseif labelrotation === :parallel
            _EDGE_ANGLES[k]
        else
            Float64(labelrotation)
        end

        for (m, t) in enumerate(tickss[k])
            p = bary2cart(_tick_bary(Val(k), t))
            ticksvisible && push!(segs, p, p + ticklength * n)
            if labelsvisible
                push!(positions, p + labeloffset * n)
                push!(strings, isnothing(labelss) ? formats[k](t) : string(labelss[k][m]))
                push!(rotations, rot)
            end
        end
    end

    isempty(segs) || linesegments!(ax, segs; color=tickcolor, linewidth=tickwidth)
    isempty(positions) || text!(ax, positions; text=strings, rotation=rotations,
        fontsize=labelsize, color=labelcolor, align=(:center, :center))
end

function _corner_labels!(ax, labels, size, color, offset)
    positions = Point2f[]
    for k in 1:3
        d = CORNERS[k] - CENTROID
        push!(positions, CORNERS[k] + offset * (d / hypot(d[1], d[2])))
    end
    text!(ax, positions; text=[string(l) for l in labels],
        fontsize=size, color=color, align=(:center, :center))
end

function _arrows!(ax, labels, color, width, headsize, offset, span,
    labelsize, labeloffset
)
    lo, hi = span
    for k in 1:3
        n = _TICK_NORMALS[k]
        b1, b2 = _arrow_bary(Val(k))
        p1 = bary2cart(b1)
        p2 = bary2cart(b2)
        d = p2 - p1
        tail = p1 + lo * d + offset * n
        head = p1 + hi * d + offset * n
        lines!(ax, [tail, head]; color, linewidth=width)
        scatter!(ax, [head]; color, marker=:utriangle, markersize=headsize,
            rotation=atan(d[2], d[1]) - pi / 2)
        text!(ax, [(tail + head) / 2 + labeloffset * n]; text=[string(labels[k])],
            fontsize=labelsize, color=color, rotation=_EDGE_ANGLES[k],
            align=(:center, :center))
    end
end

end # module
