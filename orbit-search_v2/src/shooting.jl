# Create shooting functions to find heteroclinic orbits connecting the fixed points
using LinearAlgebra, DifferentialEquations

# =============================================================================
# Shooting method to find one trajectory from a source

struct TRAJECTORY
    name::String
    source::FixedPoint
    eps::Float64
    perturbation::Vector{Float64}
    exceeded_threshold::Bool
    t::Vector{Float64}
    u::Matrix{Float64}  # each column is the state at a time point
end

function find_heteroclinic_orbit(source::FixedPoint, perturbation::Vector{Float64}, eps::Float64, name::String="", T::Float64=100.0, norm_threshold::Float64=1e2)
    # Compute unstable subspace at source
    
    coords = vcat(source.coords, zeros(3))  # (A₁, A₂, A₃, B₁, B₂, B₃)

    init_cond = coords + perturbation * eps

    p = source.params

    # Define the ODE problem and solve
    prob = ODEProblem((du,u,p,t) -> (du .= F(u,p)), init_cond, (0.0, T), p)
    sol = solve(prob, Tsit5(), saveat=0.05)

    max_norm = maximum(norm.(sol.u))

    # if max_norm > norm_threshold
    #     @warn "Solution norm exceeds threshold"
    # end

    if name == ""
        name = "orbit-from-$(source.name)"
    end

    traj = TRAJECTORY(
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

function shoot_to_most_unstable(fp::FixedPoint, eps::Float64=1e-3)
    if fp.dimension_unstable == 0
        @warn "Source fixed point has no unstable directions. No orbits to search."
        return
    else
        println("\n--- Shooting from fixed point: $(fp.name) ---")
        perturbation, _ = most_unstable_direction(fp)
        traj_plus = find_heteroclinic_orbit(fp, perturbation, eps)
        traj_minus = find_heteroclinic_orbit(fp, -perturbation, eps)
        if traj_plus.exceeded_threshold
            println("Trajectory from $(fp.name) exceeded threshold")
        else
            println("Trajectory from $(fp.name) did not exceed threshold. Create front image.")
            return traj_plus
        end
        if traj_minus.exceeded_threshold
            println("Trajectory from $(fp.name) exceeded threshold.")
        else
            println("Trajectory from $(fp.name) did not exceed threshold. Create front image.")
            return traj_minus
        end
    end
end

# =============================================================================
# Look for more interesting orbits

# Initial coarse search finding top 10 candidates for refinement later

struct OrbitScore
    trajectory::TRAJECTORY
    fp::FixedPoint
    time_near::Float64
    min_distance::Float64
end

function Base.show(io::IO, s::OrbitScore)
    println(io, "OrbitScore:")
    println(io, "  trajectory : $(s.trajectory.name)")
    println(io, "  near fp    : $(s.fp.name)")
    println(io, "  time_near  : $(round(s.time_near, digits=3))")
    println(io, "  min_dist   : $(round(s.min_distance, digits=3))")
end

# Compute score of orbit to one fixed point

function score_orbit(traj::TRAJECTORY, fp::FixedPoint, threshold::Float64=1e-2)
    min_distance = Inf
    time_near = NaN

    t = traj.t
    u = traj.u

    dists = [norm(u[:, i][1:3] - fp.coords) for i in 1:length(t)]  # Only compare A₁,A₂,A₃
    near = dists .< threshold
    dt = diff(traj.t)
    time_near = sum(dt[i] for i in eachindex(dt) if near[i] && near[i+1]; init = 0.0)
    min_distance = minimum(dists)

    return OrbitScore(traj, fp, time_near, min_distance)
end

function find_interesting_orbits_coarse(source::FixedPoint,fps::Vector{FixedPoint},eps::Float64,threshold::Float64=0.2,no_of_orbits_searched::Int=100,top_n::Int=10)
    
    candidates::Vector{FixedPoint} = find_candidate_fps(source,fps)

    scores::Vector{OrbitScore} = OrbitScore[]

    if isempty(candidates)
        @warn "No candidate fixed points with lower energy than source. No orbits to search."
        return OrbitScore[]
    end
    
    if source.dimension_unstable == 0
        @warn "Source fixed point has no unstable directions. No orbits to search."
        return scores
    end

    if source.dimension_unstable == 1
        @warn "Source fixed point has only one unstable direction. Perturbations will be along the same line."
        traj = shoot_to_most_unstable(source, eps)
        scores = [score_orbit(traj, fp, threshold) for fp in candidates]
        sort!(scores, by=s -> (-s.time_near, s.min_distance))
        return first(scores, min(top_n, length(scores)))
    end

    perturbations::Vector{Vector{Float64}} = random_perturbations(source, no_of_orbits_searched)

    i = 0

    println("\n" * "="^70)
    println("INITIATING COARSE SEARCH FROM SOURCE: $(source.name)")
    println("\n" * "="^70)

    for pert in perturbations
        i += 1
        
        if i % 1000 == 0
            println("Shooting orbit $i / $no_of_orbits_searched")
        end

        traj = find_heteroclinic_orbit(source, pert, eps, "orbit-$(i)")
        traj.exceeded_threshold && continue

        for fp in candidates
            score = score_orbit(traj, fp, threshold)
            if score.time_near >0
                println(score)
                push!(scores, score)
            end
        end
    end

    # Sort scores and return top n
    sort!(scores, by=s -> (-s.time_near, s.min_distance))
    return first(scores, min(top_n, length(scores)))
end

function find_candidate_fps(source::FixedPoint, fps::Vector{FixedPoint})
    # Return list of non-trivial fixed points with smaller energy than source
    return filter(fp -> fp.name != source.name && fp.energy < source.energy && fp.name != "trivial", fps)
end

function create_perturbations(fp::FixedPoint, n::Int)
    # Create n random perturbations in the unstable subspace
    dim_u = fp.dimension_unstable
    U = fp.unstable_subspace

    if dim_u == 0
        @warn "Fixed point has no unstable directions. No perturbations can be created."
        return Vector{Float64}[]
    
    elseif dim_u == 1
        @warn "Fixed point has unstable dimension $dim_u. Perturbations will be along the same line."
        return [most_unstable_direction(fp)[1]]
    
    elseif dim_u == 2
        angles = LinRange(0, 2π, n + 1)[1:end-1]
        return [normalize(U * [cos(θ), sin(θ)]) for θ in angles]
    
    elseif dim_u == 3
        return fibonacci_sphere(n) |> x -> [normalize(U * v) for v in x]
    
    else
        @warn "Fixed point has unstable dimension $dim_u > 3. Perturbations will be random in the subspace."
        return [normalize(U * randn(dim_u)) for _ in 1:n]
    end
end

function random_perturbations(fp::FixedPoint, n::Int)
    dim_u = fp.dimension_unstable
    U = fp.unstable_subspace

    if dim_u == 0
        @warn "Fixed point has no unstable directions. No perturbations can be created."
        return Vector{Float64}[]
    
    else
        return [normalize(U * randn(dim_u)) for _ in 1:n]
    end
end

function fibonacci_sphere(n_points::Int)
    points = Vector{Float64}[]
    golden = (1 + sqrt(5)) / 2
    for i in 0:(n_points - 1)
        phi   = 2π * i / golden
        psi   = acos(1 - 2 * (i + 0.5) / n_points)
        push!(points, [cos(phi) * sin(psi), sin(phi) * sin(psi), cos(psi)])
    end
    return points
end

function fine_search(source::FixedPoint,fps::Vector{FixedPoint},eps::Float64,threshold::Float64=0.2,no_of_orbits_searched::Int=100,top_n::Int=10)
    good_orbits = find_interesting_orbits_coarse(source, fps, eps, threshold, no_of_orbits_searched, top_n)
    scores::Vector{OrbitScore} = copy(good_orbits)

    # For each good orbit, we could implement a local optimization/refinement procedure to maximize time_near or minimize min_distance.
    for orbit in good_orbits
        println("\n" * "="^70)
        println("REFINING ORBIT: $(orbit.trajectory.name) near fp: $(orbit.fp.name)")
        println("="^70)
        i=0
        no_of_fine_search_iterations = no_of_orbits_searched  # Arbitrary choice for how many refinements to attempt per orbit
        best_score = orbit
        println("\nRefining orbit: $(orbit.trajectory.name) near fp: $(orbit.fp.name) with time_near: $(round(orbit.time_near, digits=3)) and min_distance: $(round(orbit.min_distance, digits=3))")
        U = orbit.trajectory.source.unstable_subspace
        dim_u = orbit.trajectory.source.dimension_unstable
        fp = orbit.fp
        while i < no_of_fine_search_iterations
            i += 1

            if i % 1000 == 0
                println("Shooting orbit $i / $no_of_fine_search_iterations for refinement")
            end

            new_perturbation = normalize(best_score.trajectory.perturbation + rand() * 0.1 * normalize(U * randn(dim_u)))
            new_orbit = find_heteroclinic_orbit(orbit.trajectory.source, new_perturbation, eps, "refined-$(orbit.trajectory.name)-$(i)")
            new_orbit.exceeded_threshold && continue

            score = score_orbit(new_orbit, fp, threshold)
            if score.time_near > best_score.time_near || (score.time_near >= best_score.time_near - 1e-6 && score.min_distance < best_score.min_distance)
                println("BETTER ORBIT FOUND!")
                println(score)
                push!(scores, score)
                best_score = score
            end
        end
    end
    sort!(scores, by=s -> (-s.time_near, s.min_distance))
    return first(scores, min(top_n, length(scores)))
end

function find_best_orbit(source::FixedPoint,fps::Vector{FixedPoint},eps::Float64,threshold::Float64=0.2,no_of_orbits_searched::Int=100)
    great_orbits = fine_search(source, fps, eps, threshold, no_of_orbits_searched, 2)

    scores::Vector{OrbitScore} = copy(great_orbits)

    for orbit in great_orbits
        i = 0

        println("\n" * "="^70)
        println("REFINING BEST ORBIT: $(orbit.trajectory.name) near fp: $(orbit.fp.name)")
        println("="^70)

        best_score = orbit
        U = orbit.trajectory.source.unstable_subspace
        dim_u = orbit.trajectory.source.dimension_unstable
        fp = orbit.fp

        no_of_best_orbit_iterations = no_of_orbits_searched * 10

        while i < no_of_best_orbit_iterations
            i += 1

            if i % 1000 == 0
                println("Shooting orbit $i / $no_of_best_orbit_iterations for refinement")
            end

            new_perturbation = normalize(best_score.trajectory.perturbation + rand() * 0.1 * normalize(U * randn(dim_u)))
            new_orbit = find_heteroclinic_orbit(orbit.trajectory.source, new_perturbation, eps, "refined-$(orbit.trajectory.name)-$(i)",200.0)
            new_orbit.exceeded_threshold && continue

            score = score_orbit(new_orbit, fp, threshold)
            if score.time_near > best_score.time_near || (score.time_near >= best_score.time_near - 1e-6 && score.min_distance < best_score.min_distance)
                println("BETTER ORBIT FOUND!")
                println(score)
                push!(scores, score)
                best_score = score
            end
        end
    end
    sort!(scores, by=s -> (-s.time_near, s.min_distance))
    return first(scores, min(3, length(scores)))
end