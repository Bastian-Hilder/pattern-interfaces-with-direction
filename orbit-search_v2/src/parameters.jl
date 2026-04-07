# Create default parameters and constants for the heteroclinic search

# =============================================================================
# parameters.jl  –  Parameters struct + global constants
# This module defines the `Params` struct, which encapsulates all physical
# parameters for the three-mode amplitude equations. It also defines default values and numerical
# constants used throughout the package.
# =============================================================================  

DEFAULT_K0    =  -6.0
DEFAULT_K2    =  -3.0
DEFAULT_BETA2 =  1.0
DEFAULT_MU0   =  1.0
DEFAULT_C0    =  8.0
DEFAULT_THETA =  0.0
DEFAULT_T     =  100.0

struct Params
    K0::Float64
    K2::Float64
    beta2::Float64
    mu0::Float64
    c0::Float64
    theta::Float64
    T::Float64
end

function Params(;
    K0    = DEFAULT_K0,
    K2    = DEFAULT_K2,
    beta2 = DEFAULT_BETA2,
    mu0   = DEFAULT_MU0,
    c0    = DEFAULT_C0,
    theta = DEFAULT_THETA,
    T     = DEFAULT_T
)
    p = Params(K0, K2, beta2, mu0, c0, theta, T)

    println("Creating Params with: K0=$K0, K2=$K2, beta2=$beta2, mu0=$mu0, c0=$c0, theta=$(theta*180/π)°, T=$T")
    println("Critical speed for these parameters: $(critical_speed(p))")
    println("The chosen speed is $(c0) and the critical speed is $(critical_speed(p)).")
    return p
end
# =============================================================================

function kappas(θ)
    d  = polar(θ)
    k1 = polar(0)
    k2 = polar(2π / 3)
    k3 = polar(-2π / 3)

    κ1 = -1.0 / (4 * dot(d, k1)^2)
    κ2 = -1.0 / (4 * dot(d, k2)^2)
    κ3 = -1.0 / (4 * dot(d, k3)^2)

    return k1, k2, k3, κ1, κ2, κ3
end

function polar(θ)
    return [cos(θ), sin(θ)]
end

# -----------------------------------------------------------------------------
# Critical speed check

function critical_speed(p::Params)
    k1, k2, k3, κ1, κ2, κ3 = kappas(p.theta)
    d = polar(p.theta)
    return maximum([4*abs(dot(d, k1)), 4*abs(dot(d, k2)), 4*abs(dot(d, k3))]) * sqrt(p.mu0)
end