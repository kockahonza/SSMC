# NOTE: OLD VERSION
################################################################################
# First deal with solving cubic equation...my god this took a while
################################################################################
function eval_cubic(ps, x)
    ps[1] * x^3 + ps[2] * x^2 + ps[3] * x + ps[4]
end
export eval_cubic

function find_real_cubic_roots(a::F, b::F, c::F, d::F;
    threshold=10 * eps(F)
) where {F<:AbstractFloat}
    if abs(a) < threshold
        if abs(b) < threshold
            if abs(c) < threshold # constant polynomial
                if abs(d) < threshold
                    @warn "all coefficients are close to zero"
                end
                return F[]
            else # linear polynomial
                return F[-d/c]
            end
        end # quadratic polynomial
        D = c^2 - 4 * b * d
        if D < -threshold
            return F[]
        elseif abs(D) < threshold
            return F[-c/(2*b)]
        else
            sqrt_disc = sqrt(D)
            return F[-c+sqrt_disc, -c-sqrt_disc] ./ (2 * b)
        end
    else
        return cardano(a, b, c, d; threshold)
    end
end
find_real_cubic_roots(ps::SVector{4}; kwargs...) = find_real_cubic_roots(ps[1], ps[2], ps[3], ps[4]; kwargs...)
function cardano(a::F, b::F, c::F, d::F;
    threshold=10 * eps(F)
) where {F}
    # get rid of a as it could still be very small and we'd need to divide by a^3
    bp = b / a
    cp = c / a
    dp = d / a

    p = (3 * cp - bp^2) / 3
    q = (2 * bp^3 - 9 * bp * cp + 27 * dp) / 27

    D = q^2 / 4 + p^3 / 27
    minushalfq = -q / 2

    if D > threshold # non-zero, positive D - guaranteed 1 real solution
        sqrtD = sqrt(D)
        u1 = minushalfq + sqrtD
        u2 = minushalfq - sqrtD
        t_roots = F[cbrt(u1)+cbrt(u2)]
    else
        Dnot0 = D < -threshold
        if Dnot0 # non-zero, negative D - guaranteed 3 real solutions, but we need complex numbers to get to them
            xx = minushalfq + im * sqrt(-D)
        else # D ~ 0 we get repeated roots
            xx = complex(minushalfq)
        end

        C1 = xx^(1 / 3) # any cube root should work here
        third_angle = exp((2 * pi / 3) * im)
        C2 = C1 * third_angle
        C3 = C2 * third_angle

        t1 = real(C1 - p / (3 * C1))
        t2 = real(C2 - p / (3 * C2))
        t3 = real(C3 - p / (3 * C3))

        if Dnot0 # this is mostly for speed, no need to deal with isapprox when D !~ 0
            t_roots = F[t1, t2, t3]
        else
            eq12 = isapprox(t1, t2; rtol=threshold)
            eq13 = isapprox(t1, t3; rtol=threshold)
            eq23 = isapprox(t2, t3; rtol=threshold)

            if (eq12 + eq13 + eq23) > 1 # all roots are equal (allowing for funky transitivity)
                t_roots = F[t1]
            elseif eq12
                t_roots = F[t1, t3]
            elseif eq13
                t_roots = F[t1, t2]
            elseif eq23
                t_roots = F[t2, t3]
            else
                t_roots = F[t1, t2, t3]
            end
        end
    end

    return t_roots .- (bp / 3)
end
export find_real_cubic_roots

# This is the one we actually want
function find_real_positive_cubic_roots(args...; kwargs...)
    real_roots = find_real_cubic_roots(args...; kwargs...)
    typed_zero = zero(eltype(real_roots))
    filter(x -> x > typed_zero, real_roots)
end
export find_real_positive_cubic_roots

"""This is hardcoded for a 3 by 3 matrix M1!!"""
function make_K_polynomial(M1, Ds)
    a = M1[1, 1]
    b = M1[1, 2]
    c = M1[1, 3]
    d = M1[2, 1]
    e = M1[2, 2]
    f = M1[2, 3]
    g = M1[3, 1]
    h = M1[3, 2]
    i = M1[3, 3]
    o = Ds[1]
    p = Ds[2]
    q = Ds[3]

    t0 = a * e * i - a * f * h - b * d * i + b * f * g + c * d * h - c * e * g
    t1 = -a * e * q - a * i * p + b * d * q + c * g * p - e * i * o + f * h * o
    t2 = a * p * q + e * o * q + i * o * p
    t3 = -o * p * q

    SA[t3, t2, t1, t0]
end
export make_K_polynomial

