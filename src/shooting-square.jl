# Create shooting functions for SQUARE LATTICE to find heteroclinic orbits
using LinearAlgebra, DifferentialEquations

# Helper: check if theta is effectively zero
function is_theta_zero(theta::Float64)
    return abs(theta) < 1e-10
end

# =============================================================================
# Shooting method for square lattice

struct TRAJECTORY_SQUARE
    name::String
    source::FixedPointSquare
    eps::Float64
    perturbation::Vector{Float64}
    exceeded_threshold::Bool
    t::Vector{Float64}
    u::Matrix{Float64}  # each column is the state at a time point
end

function find_heteroclinic_orbit_square(source::FixedPointSquare, perturbation::Vector{Float64}, eps::Float64, name::String="", T::Float64=100.0; norm_threshold::Float64=1e2)
    # Compute unstable subspace at source
    p = source.params
    
    if is_theta_zero(p.theta)
        # Special case: θ = 0, state is (A₁, B₁, A₂)
        coords = [source.coords[1], 0.0, source.coords[2]]  # (A₁, B₁, A₂)
    else
        # Standard case: state is (A₁, A₂, B₁, B₂)
        coords = vcat(source.coords, zeros(2))  # (A₁, A₂, B₁, B₂)
    end

    init_cond = coords + perturbation * eps

    # Define the ODE problem and solve
    # Use stiff solver for θ=0 case due to fast relaxation
    if is_theta_zero(p.theta)
        prob = ODEProblem((du,u,p,t) -> (du .= F_square(u,p)), init_cond, (0.0, T), p)
        sol = solve(prob, Rosenbrock23(), saveat=0.05, reltol=1e-6, abstol=1e-8)
    else
        prob = ODEProblem((du,u,p,t) -> (du .= F_square(u,p)), init_cond, (0.0, T), p)
        sol = solve(prob, Tsit5(), saveat=0.05)
    end

    max_norm = maximum(norm.(sol.u))

    if name == ""
        name = "orbit-from-$(source.name)"
    end

    traj = TRAJECTORY_SQUARE(
        name,
        source,
        eps,
        perturbation,
        max_norm > norm_threshold,
        sol.t,
        hcat(sol.u...)  # Convert array of vectors to matrix
    )

    return traj
end

# =============================================================================
# Shoot into most unstable direction

function shoot_to_most_unstable_square(fp::FixedPointSquare, eps::Float64=1e-3)
    if fp.dimension_unstable == 0
        @warn "Source fixed point has no unstable directions. No orbits to search."
        return nothing
    end

    println("\n--- Shooting from fixed point (square): $(fp.name) ---")
    perturbation, _ = most_unstable_direction_square(fp)

    delta = norm(fp.coords) > 0 ? 0.5 * norm(fp.coords) : 0.05

    # Adaptive threshold: cap at 0.05, but tighten proportionally when eps is small
    near_zero_threshold = min(0.05, delta)

    for (label, direction) in [("+", perturbation), ("-", -perturbation)]
        traj = find_heteroclinic_orbit_square(fp, direction, eps)
        end_norm = norm(traj.u[:, end])
        println("  Direction $label: final ‖u‖ = $(round(end_norm, sigdigits=4)) (threshold = $near_zero_threshold)")
        if end_norm < near_zero_threshold
            println("  → Converged to trivial state.")
            return traj
        end
    end

    println("  Neither direction converged to trivial state.")
    return nothing
end

# =============================================================================
# Look for more interesting orbits

struct OrbitScoreSquare
    trajectory::TRAJECTORY_SQUARE
    fp::FixedPointSquare
    time_near::Float64
    min_distance::Float64
end

function Base.show(io::IO, s::OrbitScoreSquare)
    println(io, "OrbitScoreSquare:")
    println(io, "  trajectory : $(s.trajectory.name)")
    println(io, "  near fp    : $(s.fp.name)")
    println(io, "  time_near  : $(round(s.time_near, digits=3))")
    println(io, "  min_dist   : $(round(s.min_distance, digits=3))")
end

# Compute score of orbit to one fixed point

function score_orbit_square(traj::TRAJECTORY_SQUARE, fp::FixedPointSquare, threshold::Float64=1e-2)
    min_distance = Inf
    time_near = NaN

    t = traj.t
    u = traj.u

    # Extract A₁, A₂ components (first two rows regardless of state dimension)
    dists = [norm(u[1:2, i] - fp.coords) for i in 1:length(t)]  # Only compare A₁, A₂
    near = dists .< threshold
    dt = diff(traj.t)
    time_near = sum(dt[i] for i in eachindex(dt) if near[i] && near[i+1]; init = 0.0)
    min_distance = minimum(dists)

    return OrbitScoreSquare(traj, fp, time_near, min_distance)
end

function find_candidate_fps_square(source::FixedPointSquare, fps::Vector{FixedPointSquare})
    # Return list of non-trivial fixed points with smaller energy than source
    return filter(fp -> fp.name != source.name && fp.energy < source.energy && fp.name != "trivial", fps)
end

function random_perturbations_square(fp::FixedPointSquare, n::Int)
    dim_u = fp.dimension_unstable
    U = fp.unstable_subspace

    if dim_u == 0
        @warn "Fixed point has no unstable directions. No perturbations can be created."
        return Vector{Float64}[]
    else
        return [normalize(U * randn(dim_u)) for _ in 1:n]
    end
end
