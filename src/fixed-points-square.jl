# Create everything necessary to analyse the fixed points for SQUARE LATTICE

# =============================================================================
# FIXED POINTS - SQUARE LATTICE
# =============================================================================

# Helper: check if theta is effectively zero
function is_theta_zero(theta::Float64)
    return abs(theta) < 1e-10
end

struct FixedPointSquare
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
function FixedPointSquare(name::String, coords::Vector{Float64}, params::Params)
    J = compute_jacobian_square(coords, params)
    λs, vs = eigen(J) # eigenvalues and eigenvectors
    unstable_idx = findall(real.(λs) .>  1e-10)
    stable_idx   = findall(real.(λs) .< -1e-10)
    unstable_subspace = real.(vs[:, unstable_idx])
    stable_subspace   = real.(vs[:, stable_idx])
    unstable_eigenvalues = λs[unstable_idx]
    stable_eigenvalues   = λs[stable_idx]
    dimension_unstable = length(unstable_idx)
    dimension_stable   = length(stable_idx)
    
    # Construct appropriate state vector for energy calculation
    if is_theta_zero(params.theta)
        state = [coords[1], 0.0, coords[2]]  # (A₁, B₁, A₂) - matches F_square ordering
    else
        state = vcat(coords, zeros(2))  # (A₁, A₂, B₁, B₂)
    end
    energy = lyapunov_square(state, params)
    
    return FixedPointSquare(name, coords, unstable_subspace, stable_subspace, unstable_eigenvalues, dimension_unstable, stable_eigenvalues, dimension_stable, energy, params)
end

# =============================================================================

# For θ = 0, A₂ satisfies algebraic equation: μ₀ A₂ + (K₀|A₂|² + K₂|A₁|²) A₂ = 0
function solve_A2_algebraic(A1::Float64, p::Params)
    # A₂ = 0 or A₂² = -(μ₀ + K₂|A₁|²) / K₀
    if p.mu0 + p.K2 * A1^2 >= 0
        return 0.0  # Only trivial solution
    else
        # Non-trivial solution exists
        return sqrt(-(p.mu0 + p.K2 * A1^2) / p.K0)
    end
end

function F_square(state::AbstractVector, p::Params)
    _, _, κ1, κ2 = kappas_square(p.theta)
    
    if is_theta_zero(p.theta)
        # Special case: θ = 0, A₂ is algebraic
        A1, B1, A2 = state
        
        dA1_dt = B1
        dB1_dt = κ1 * (p.mu0 * A1 + p.K0 * A1^3 + p.K2 * A1 * A2^2) + κ1 * p.c0 * B1
        
        # Algebraic constraint: c₀ dA₂/dξ = -μ₀ A₂ - (K₀|A₂|² + K₂|A₁|²) A₂
        # For ODE solver, we implement this as fast relaxation:
        dA2_dt = -(p.mu0 * A2 + p.K0 * A2^3 + p.K2 * A2 * A1^2) / p.c0
        
        return [dA1_dt, dB1_dt, dA2_dt]
    else
        # Standard case: both modes propagate
        A1, A2, B1, B2 = state
        
        dA1_dt = B1
        dA2_dt = B2
        
        dB1_dt = κ1 * (p.mu0 * A1 + p.K0 * A1^3 + p.K2 * A1 * A2^2) + κ1 * p.c0 * B1
        dB2_dt = κ2 * (p.mu0 * A2 + p.K0 * A2^3 + p.K2 * A2 * A1^2) + κ2 * p.c0 * B2
        
        return [dA1_dt, dA2_dt, dB1_dt, dB2_dt]
    end
end

function compute_jacobian_square(A::Vector{Float64}, p::Params)
    _, _, κ1, κ2 = kappas_square(p.theta)
    
    if is_theta_zero(p.theta)
        # Special case: θ = 0, 3D system (A1, B1, A2)
        J = zeros(3, 3)
        A1 = A[1]
        A2 = length(A) >= 2 ? A[2] : 0.0
        
        # dA1/dt = B1
        J[1, 2] = 1.0
        
        # dB1/dt
        J[2, 1] = κ1 * (p.mu0 + 3 * p.K0 * A1^2 + p.K2 * A2^2)
        J[2, 2] = κ1 * p.c0
        J[2, 3] = κ1 * (2 * p.K2 * A1 * A2)
        
        # dA2/dt (algebraic, fast relaxation)
        J[3, 1] = -(2 * p.K2 * A2 * A1) / p.c0
        J[3, 3] = -(p.mu0 + 3 * p.K0 * A2^2 + p.K2 * A1^2) / p.c0
        
        return J
    else
        # Standard case: 4D system
        J = zeros(4, 4)
        
        # dA/dt = A'  (identity block)
        J[1:2, 3:4] = I(2)
        
        # d(A₁'')/d(A₁,A₂,A₁')
        J[3, 1] = κ1 * (p.mu0 + 3 * p.K0 * A[1]^2 + p.K2 * A[2]^2)
        J[3, 2] = κ1 * (2 * p.K2 * A[1] * A[2])
        J[3, 3] = κ1 * p.c0
        
        # d(A₂'')/d(A₁,A₂,A₂')
        J[4, 1] = κ2 * (2 * p.K2 * A[2] * A[1])
        J[4, 2] = κ2 * (p.mu0 + 3 * p.K0 * A[2]^2 + p.K2 * A[1]^2)
        J[4, 4] = κ2 * p.c0
        
        return J
    end
end

# -----------------------------------------------------------------------------

function lyapunov_square(state::AbstractVector, p::Params)
    _, _, κ1, κ2 = kappas_square(p.theta)
    
    if is_theta_zero(p.theta)
        # Special case: θ = 0, 3D state (A1, B1, A2)
        A1, B1, A2 = state
        
        # Only A1 mode has kinetic energy
        kinetic = (-2/κ1) * B1^2
        
        potential = (p.mu0/2) * (A1^2 + A2^2) +
                    (p.K0/4) * (A1^4 + A2^4) +
                    (p.K2/2) * (A1^2 * A2^2)
        
        return kinetic + potential
    else
        # Standard case
        A1, A2, B1, B2 = state
        
        kinetic = (-2/κ1) * B1^2 + (-2/κ2) * B2^2
        
        potential = (p.mu0/2) * (A1^2 + A2^2) +
                    (p.K0/4) * (A1^4 + A2^4) +
                    (p.K2/2) * (A1^2 * A2^2)
        
        return kinetic + potential
    end
end

# -----------------------------------------------------------------------------

function print_fixed_point_square(fp::FixedPointSquare)
    println("Fixed Point (Square): $(fp.name)")
    println("  Coordinates: $(fp.coords)")
    println("  Energy: $(fp.energy)")
    println("  Unstable eigenvalues: $(fp.unstable_eigenvalues)")
    println("  Stable eigenvalues: $(fp.stable_eigenvalues)")
    println("  Dimension of unstable subspace: $(fp.dimension_unstable)")
    println("  Dimension of stable subspace: $(fp.dimension_stable)")
end

# -----------------------------------------------------------------------------
# Instantiate the fixed points for the set of parameters we want to analyze

function instantiate_fixed_points_square(p::Params)
    fps = FixedPointSquare[]
    # Trivial fixed point
    push!(fps, trivial_fp_square(p))
    # Roll waves (always exist when K0*mu0 < 0)
    if p.K0*p.mu0 < 0
        push!(fps, roll_wave_fp_square(p))
    end
    # Squares - existence and energy selection depends on K0 vs K2
    if p.mu0 * (p.K0 + p.K2) < 0
        push!(fps, square_fp(p))
    end
    return fps
end

function trivial_fp_square(p::Params)
    return FixedPointSquare("trivial", [0.0, 0.0], p)
end

function roll_wave_fp_square(p::Params)
    A1 = sqrt(-p.mu0 / p.K0)
    return FixedPointSquare("roll_wave", [A1, 0.0], p)
end

function square_fp(p::Params)
    A = sqrt(-p.mu0 / (p.K0 + p.K2))
    return FixedPointSquare("square", [A, A], p)
end

function square_fold_point(p::Params)
    mu = 0.0  # Squares bifurcate from trivial at mu0 = 0 when K0 + K2 < 0
    return mu
end

# -----------------------------------------------------------------------------
# Get most unstable direction

function most_unstable_direction_square(fp::FixedPointSquare)
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

function write_energy_square(fp_factory::Function, p::Params, mumin::Float64, mumax::Float64, step_size::Float64, filename::String)
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
