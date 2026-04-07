# Create everything necessary to analyse the fixed points

# =============================================================================
# FIXED POINTS
# =============================================================================

struct FixedPoint
    name::String
    coords::Vector{Float64}
    unstable_subspace::Matrix{Float64}
    stable_subspace::Matrix{Float64}
    unstable_eigenvalues::Vector{ComplexF64}
    dimension_unstable::Int
    stable_eigenvalues::Vector{ComplexF64}
    dimension_stable::Int
    energy::Float64
    params::Params
end

# Convenience constructor: compute unstable and stable subspaces automatically
function FixedPoint(name::String, coords::Vector{Float64}, params::Params)
    J = compute_jacobian(coords, params)
    λs, vs = eigen(J) # eigenvalues and eigenvectors
    unstable_idx = findall(real.(λs) .>  1e-10)
    stable_idx   = findall(real.(λs) .< -1e-10)
    unstable_subspace = real.(vs[:, unstable_idx])
    stable_subspace   = real.(vs[:, stable_idx])
    unstable_eigenvalues = λs[unstable_idx]
    stable_eigenvalues   = λs[stable_idx]
    dimension_unstable = length(unstable_idx)
    dimension_stable   = length(stable_idx)
    energy = lyapunov(vcat(coords, zeros(3)), params)
    return FixedPoint(name, coords, unstable_subspace, stable_subspace, unstable_eigenvalues, dimension_unstable, stable_eigenvalues, dimension_stable, energy, params)
end

# =============================================================================

function F(state::AbstractVector, p::Params)
    A1, A2, A3, B1, B2, B3 = state
    _, _, _, κ1, κ2, κ3 = kappas(p.theta)

    dA1_dt = B1
    dA2_dt = B2
    dA3_dt = B3

    dB1_dt = κ1 * (p.mu0 * A1 + p.beta2 * A2 * A3 + p.K0 * A1^3 + p.K2 * A1 * (A2^2 + A3^2)) + κ1 * p.c0 * B1
    dB2_dt = κ2 * (p.mu0 * A2 + p.beta2 * A1 * A3 + p.K0 * A2^3 + p.K2 * A2 * (A1^2 + A3^2)) + κ2 * p.c0 * B2
    dB3_dt = κ3 * (p.mu0 * A3 + p.beta2 * A1 * A2 + p.K0 * A3^3 + p.K2 * A3 * (A1^2 + A2^2)) + κ3 * p.c0 * B3

    return [dA1_dt, dA2_dt, dA3_dt, dB1_dt, dB2_dt, dB3_dt]
end

function compute_jacobian(A::Vector{Float64}, p::Params)
    _, _, _, κ1, κ2, κ3 = kappas(p.theta)

    J = zeros(6, 6)

    # dA/dt = A'  (identity block)
    J[1:3, 4:6] = I(3)

    # d(A₁'')/d(A₁,A₂,A₃,A₁')
    J[4, 1] = κ1 * (p.mu0 + 3 * p.K0 * A[1]^2 + p.K2 * (A[2]^2 + A[3]^2))
    J[4, 2] = κ1 * (p.beta2 * A[3] + 2 * p.K2 * A[1] * A[2])
    J[4, 3] = κ1 * (p.beta2 * A[2] + 2 * p.K2 * A[1] * A[3])
    J[4, 4] = κ1 * p.c0

    # d(A₂'')/d(A₁,A₂,A₃,A₂')
    J[5, 1] = κ2 * (p.beta2 * A[3] + 2 * p.K2 * A[2] * A[1])
    J[5, 2] = κ2 * (p.mu0 + 3 * p.K0 * A[2]^2 + p.K2 * (A[1]^2 + A[3]^2))
    J[5, 3] = κ2 * (p.beta2 * A[1] + 2 * p.K2 * A[2] * A[3])
    J[5, 5] = κ2 * p.c0

    # d(A₃'')/d(A₁,A₂,A₃,A₃')
    J[6, 1] = κ3 * (p.beta2 * A[2] + 2 * p.K2 * A[3] * A[1])
    J[6, 2] = κ3 * (p.beta2 * A[1] + 2 * p.K2 * A[3] * A[2])
    J[6, 3] = κ3 * (p.mu0 + 3 * p.K0 * A[3]^2 + p.K2 * (A[1]^2 + A[2]^2))
    J[6, 6] = κ3 * p.c0

    return J
end

function eigen(J::Matrix{Float64})
    # Compute eigenvalues and eigenvectors, ensuring they are complex if necessary
    result = LinearAlgebra.eigen(J)
    return result.values, result.vectors
end

# -----------------------------------------------------------------------------

function lyapunov(state::AbstractVector, p::Params)
    A1, A2, A3, B1, B2, B3 = state
    _, _, _, κ1, κ2, κ3 = kappas(p.theta)

    # (d·kⱼ)² = −1/κⱼ  (κⱼ < 0),  so the kinetic coefficient 2(d·kⱼ)² = −2/κⱼ > 0
    kinetic   = (-2/κ1) * B1^2 + (-2/κ2) * B2^2 + (-2/κ3) * B3^2

    potential = (p.mu0/2) * (A1^2 + A2^2 + A3^2) +
                p.beta2 * A1 * A2 * A3 +
                (p.K0/4) * (A1^4 + A2^4 + A3^4) +
                (p.K2/2) * (A1^2 * A2^2 + A1^2 * A3^2 + A2^2 * A3^2)

    return kinetic + potential
end

# -----------------------------------------------------------------------------

function print_fixed_point(fp::FixedPoint)
    println("Fixed Point: $(fp.name)")
    println("  Coordinates: $(fp.coords)")
    println("  Energy: $(fp.energy)")
    println("  Unstable eigenvalues: $(fp.unstable_eigenvalues)")
    println("  Stable eigenvalues: $(fp.stable_eigenvalues)")
    println("  Dimension of unstable subspace: $(fp.dimension_unstable)")
    println("  Dimension of stable subspace: $(fp.dimension_stable)")
end

# -----------------------------------------------------------------------------
# Instantiate the fixed points for the set of parameters we want to analyze

function instantiate_fixed_points(p::Params)
    fps = FixedPoint[]
    # Trivial fixed point
    push!(fps, trivial_fp(p))
    # Roll waves
    if p.K0*p.mu0 < 0
        push!(fps, roll_wave_fp(p))
    end
    # Hexagons
    if p.beta2^2 > 4*p.mu0*(p.K0 + 2*p.K2) && p.K0 + 2*p.K2 != 0
        push!(fps, hexagon_fp(p))
    end
    # down-hexagons
    if p.beta2^2 > 4*p.mu0*(p.K0 + 2*p.K2) && p.K0 + 2*p.K2 != 0
        push!(fps, down_hexagon_fp(p))
    end
    # Mixed modes
    if (p.K0*p.beta2^2 +(p.K0 - p.K2)^2*p.mu0)*(p.K0 + p.K2) < 0
        push!(fps, mixed_mode_fp(p))
    end
    return fps
end

function trivial_fp(p::Params)
    return FixedPoint("trivial", [0.0, 0.0, 0.0], p)
end

function roll_wave_fp(p::Params)
    A1 = sqrt(-p.mu0 / p.K0)
    return FixedPoint("roll_wave", [A1, 0.0, 0.0], p)
end

function hexagon_fp(p::Params)
    A_minus = (-p.beta2 - sqrt(p.beta2^2 - 4*p.mu0*(p.K0 + 2*p.K2))) / (2*(p.K0 + 2*p.K2))
    A_plus  = (-p.beta2 + sqrt(p.beta2^2 - 4*p.mu0*(p.K0 + 2*p.K2))) / (2*(p.K0 + 2*p.K2))
    if A_minus > 0
        return FixedPoint("hexagon", [A_minus, A_minus, A_minus], p)
    else
        return FixedPoint("hexagon", [A_plus, A_plus, A_plus], p)
    end
end

function hexagon_fold_point(p::Params)
    mu = p.beta2^2 / (4*(p.K0 + 2*p.K2))
    return mu
end

function down_hexagon_fp(p::Params)
    A_minus = (-p.beta2 - sqrt(p.beta2^2 - 4*p.mu0*(p.K0 + 2*p.K2))) / (2*(p.K0 + 2*p.K2))
    A_plus  = (-p.beta2 + sqrt(p.beta2^2 - 4*p.mu0*(p.K0 + 2*p.K2))) / (2*(p.K0 + 2*p.K2))
    if A_minus < 0
        return FixedPoint("down_hexagon", [A_minus, A_minus, A_minus], p)
    else
        return FixedPoint("down_hexagon", [A_plus, A_plus, A_plus], p)
    end
end

function mixed_mode_fp(p::Params)
    A_1 = p.beta2 / (p.K0-p.K2)
    A_23 = sqrt(-(p.K0*p.beta2^2 + (p.K0-p.K2)^2*p.mu0) / ((p.K0+p.K2)*(p.K0-p.K2)^2) )
    return FixedPoint("mixed_mode", [A_1, A_23, A_23], p)
end

function mixed_mode_bif_point(p::Params)
    mu = - (p.K0*p.beta2^2) / ((p.K0-p.K2)^2)
    return mu
end

# -----------------------------------------------------------------------------
# Get most unstable direction

function most_unstable_direction(fp::FixedPoint)
    if fp.dimension_unstable == 0
        @warn "Fixed point $(fp.name) is stable, no unstable direction."
        return nothing, 0.0
    else
        idx = argmax(real.(fp.unstable_eigenvalues))
        return fp.unstable_subspace[:, idx], fp.unstable_eigenvalues[idx]
    end
end

# -----------------------------------------------------------------------------
# Write Lyapunov energy vs mu0 to a file readable by pgfplots (\addplot table)

function write_energy(fp_factory::Function, p::Params, mumin::Float64, mumax::Float64, step_size::Float64, filename::String)
    step_size > 0 || throw(ArgumentError("step_size must be positive"))
    open(filename, "w") do f
        println(f, "mu0 energy")
        for mu0 in mumin:step_size:mumax
            p_mu = Params(p.K0, p.K2, p.beta2, mu0, p.c0, p.theta, p.T)
            try
                fp = fp_factory(p_mu)
                println(f, "$mu0 $(fp.energy)")
            catch
            end
        end
    end
end

