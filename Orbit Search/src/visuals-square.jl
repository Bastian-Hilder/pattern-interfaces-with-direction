# Create visualizations for SQUARE LATTICE fixed points and heteroclinic orbits
using Plots, Interpolations, Measures

# Helper: check if theta is effectively zero
function is_theta_zero(theta::Float64)
    return abs(theta) < 1e-10
end

# Use same quality settings as hexagonal
HIGH_QUALITY = true

DEFAULT_HEIGHT      = HIGH_QUALITY ? 2160 : 1080
DEFAULT_WIDTH       = HIGH_QUALITY ? 3840 : 1920
DEFAULT_DOMAIN_SIZE = 4π
DEFAULT_DOMAIN_SIZE_FRONT = 20π
DOMAIN_SIZE_FRONT_WIDE = 40π

struct visualization_params_square
    HEIGHT::Int
    WIDTH::Int
    COLOR_MAP::Symbol
    domain_size::Float64
end

const VIS_PARAMS_SQUARE = visualization_params_square(DEFAULT_HEIGHT, DEFAULT_WIDTH, :jet, DEFAULT_DOMAIN_SIZE)
const VIS_PARAMS_FRONT_SQUARE = visualization_params_square(DEFAULT_HEIGHT, DEFAULT_WIDTH, :jet, DEFAULT_DOMAIN_SIZE_FRONT)
const VIS_PARAMS_FRONT_WIDE_SQUARE = visualization_params_square(DEFAULT_HEIGHT, DEFAULT_WIDTH, :jet, DOMAIN_SIZE_FRONT_WIDE)

# =============================================================================
# Wavefield for square lattice (2 modes at 90°)

function wavefield_value_square(A1::Real, A2::Real, x::Real, y::Real)
    k1 = polar(0.0)
    k2 = polar(π/2)
    r  = [x, y]
    return 2.0 * (A1 * cos(dot(k1, r)) + A2 * cos(dot(k2, r)))
end

# =============================================================================
# Interpolation helper

function make_interpolator_square(t::Vector{Float64}, A::Vector{Float64})
    t_min, t_max = extrema(t)
    left_val  = A[1]
    right_val = A[end]
    itp = LinearInterpolation(t, A, extrapolation_bc=Flat())
    return xi -> xi < t_min ? left_val : xi > t_max ? right_val : itp(xi)
end

function field_moving_square(A1_itp, A2_itp, p::Params, x, y, t; rotated::Bool=false)
    k1, k2, _, _ = kappas_square(p.theta; rotated=rotated)
    d = rotated ? polar(0) : polar(p.theta)

    X = repeat(x',      length(y), 1)
    Y = repeat(y,  1,   length(x))

    # Time coordinate in the co-moving frame
    t_sample = (d[1] .* X .+ d[2] .* Y) .- p.c0 .* t
    F = 2 * (A1_itp.(t_sample) .* cos.(k1[1] .* X .+ k1[2] .* Y) +
         A2_itp.(t_sample) .* cos.(k2[1] .* X .+ k2[2] .* Y))

    return F
end

# Signed power transform
signed_power_square(z::AbstractMatrix, gamma::Float64) =
    gamma == 1.0 ? z : sign.(z) .* abs.(z) .^ gamma

# =============================================================================
# Create images of all fixed points for square lattice

function create_all_fixed_point_images_square(fps::Vector{FixedPointSquare}, run_info::RunInfo, vis_param::visualization_params_square=VIS_PARAMS_SQUARE; color_bar::Bool=true)
    println("\n" * "="^70)
    println("CREATING FIXED-POINT IMAGES (SQUARE)")
    println("="^70)
    mkpath(joinpath(run_info.folder, "fixed_points_square"))
    for fp in fps
        create_fixed_point_image_square(fp; filename=joinpath(run_info.folder, "fixed_points_square", "$(fp.name)_$(fp.params.K0)_$(fp.params.K2)_$(fp.params.mu0).png"), vis_param=vis_param, color_bar)
    end
end

function create_fp_images_square(p::Params, mumin::Float64, mumax::Float64, step_size::Float64, run_info::RunInfo, vis_param::visualization_params_square=VIS_PARAMS_SQUARE; color_bar::Bool=true)
    step_size > 0 || throw(ArgumentError("step_size must be positive"))
    for mu0 in mumin:step_size:mumax
        p_mu = Params(p.K0, p.K2, p.beta2, mu0, p.c0, p.theta, p.T)
        try
            fps = instantiate_fixed_points_square(p_mu)
            create_all_fixed_point_images_square(fps, run_info, vis_param, color_bar=color_bar)
         catch error
            @warn "Failed to create fixed point images (square) for mu0 = $mu0: $error"
        end
    end
end 

function create_fixed_point_image_square(fp::FixedPointSquare; filename::String, vis_param::visualization_params_square=VIS_PARAMS_SQUARE, color_bar::Bool=true)
    println("\nCreating image for fixed point (square): $(fp.name)")
    A1, A2 = fp.coords
    d  = vis_param.domain_size
    nx = vis_param.HEIGHT # Use HEIGHT for both dimensions to ensure square aspect ratio
    ny = vis_param.HEIGHT
    x = LinRange(-d, d, nx)
    y = LinRange(-d, d, ny)
    z = [wavefield_value_square(A1, A2, Float64(xi), Float64(yi)) for yi in y, xi in x]
    zmax = maximum(abs.(z))
    zmax = iszero(zmax) ? 1.0 : zmax  # Avoid division by zero for trivial fixed point
    plt = heatmap(
        collect(x), collect(y), z;
        color        = vis_param.COLOR_MAP,
        aspect_ratio = :equal,
        axis         = false,
        colorbar     = color_bar,
        size         = (nx, ny),
        margin       = 0mm,
        clim=(-zmax, zmax),
    )
    savefig(plt, filename)
    println("  Saved → $filename")
end

# =============================================================================
# Trajectory visualization

function create_front_image_full_square(traj::TRAJECTORY_SQUARE, run_info::RunInfo,
                                       vis_param::visualization_params_square=VIS_PARAMS_FRONT_SQUARE;
                                       color_bar::Bool=true,
                                       rotated::Bool=false, gamma::Float64=1.0,
                                       margin::Float64=2π, aspect_ratio::Int=5,
                                       time_stamp::Float64=0.0)
    println("\nCreating front image for trajectory: $(traj.name)")

    p = traj.source.params
    
    # Extract amplitudes based on state dimensionality
    if size(traj.u, 1) == 3
        # State is (A1, B1, A2)
        A1_itp = make_interpolator_square(traj.t, traj.u[1, :])
        A2_itp = make_interpolator_square(traj.t, traj.u[3, :])
    else
        # State is (A1, A2, B1, B2)
        A1_itp = make_interpolator_square(traj.t, traj.u[1, :])
        A2_itp = make_interpolator_square(traj.t, traj.u[2, :])
    end

    t_min, t_max = extrema(traj.t)
    x_left  = t_min - margin
    x_right = t_max + margin
    dw      = (x_right - x_left) / 2.0

    dh  = dw / aspect_ratio
    ny  = vis_param.HEIGHT
    nx  = ny * aspect_ratio

    x = LinRange(x_left, x_right, nx)
    y = LinRange(-dh, dh, ny)
 
    z    = signed_power_square(field_moving_square(A1_itp, A2_itp, p, x, y, time_stamp; rotated=rotated), gamma)
    zmax = maximum(abs.(z))
    zmax = iszero(zmax) ? 1.0 : zmax
 
    plt = heatmap(
        collect(x), collect(y), z;
        color        = vis_param.COLOR_MAP,
        aspect_ratio = :equal,
        axis         = false,
        colorbar     = color_bar,
        size         = (nx, ny),
        margin       = 0mm,
        clim         = (-zmax, zmax),
    )
 
    filename = joinpath(run_info.folder, "fronts_square", "$(traj.name).png")
    mkpath(dirname(filename))
    savefig(plt, filename)
    println("  Saved → $filename")
end

function create_front_image_shifted_square(traj::TRAJECTORY_SQUARE, run_info::RunInfo, shift::Float64=0.5,
                                          vis_param::visualization_params_square=VIS_PARAMS_FRONT_SQUARE;
                                          color_bar::Bool=true,
                                          rotated::Bool=false, gamma::Float64=1.0,
                                          margin::Float64=2π, aspect_ratio::Int=5,
                                          time_stamp::Float64=0.0)
    println("\nCreating shifted front image for trajectory: $(traj.name)")

    p = traj.source.params
    
    # Extract amplitudes based on state dimensionality
    if size(traj.u, 1) == 3
        # State is (A1, B1, A2)
        A1_itp = make_interpolator_square(traj.t, traj.u[1, :])
        A2_itp = make_interpolator_square(traj.t, traj.u[3, :])
    else
        # State is (A1, A2, B1, B2)
        A1_itp = make_interpolator_square(traj.t, traj.u[1, :])
        A2_itp = make_interpolator_square(traj.t, traj.u[2, :])
    end

    t_min, t_max = extrema(traj.t)
    orbit_length = t_max - t_min
    # Positive shift moves the window leftward, revealing more of the patterned (left) side
    x_left  = t_min - margin - shift * orbit_length
    x_right = t_max + margin - shift * orbit_length
    dw      = (x_right - x_left) / 2.0

    dh  = dw / aspect_ratio
    ny  = vis_param.HEIGHT
    nx  = ny * aspect_ratio

    x = LinRange(x_left, x_right, nx)
    y = LinRange(-dh, dh, ny)

    z    = signed_power_square(field_moving_square(A1_itp, A2_itp, p, x, y, time_stamp; rotated=rotated), gamma)
    zmax = maximum(abs.(z))
    zmax = iszero(zmax) ? 1.0 : zmax

    plt = heatmap(
        collect(x), collect(y), z;
        color        = vis_param.COLOR_MAP,
        aspect_ratio = :equal,
        axis         = false,
        colorbar     = color_bar,
        size         = (nx, ny),
        margin       = 0mm,
        clim         = (-zmax, zmax),
    )

    filename = joinpath(run_info.folder, "fronts_square", "$(traj.name)_shifted.png")
    mkpath(dirname(filename))
    savefig(plt, filename)
    println("  Saved → $filename")
end

function create_front_image_square(traj::TRAJECTORY_SQUARE, run_info::RunInfo, time_stamp::Float64=0.0,
                                  vis_param::visualization_params_square=VIS_PARAMS_FRONT_SQUARE;
                                  color_bar::Bool=true,
                                  rotated::Bool=false, gamma::Float64=1.0,
                                  margin::Float64=2π, aspect_ratio::Int=5)
    println("\nCreating front image for trajectory: $(traj.name)")
    
    # Extract amplitudes based on state dimensionality
    if size(traj.u, 1) == 3
        # State is (A1, B1, A2)
        A1_itp = make_interpolator_square(traj.t, traj.u[1, :])
        A2_itp = make_interpolator_square(traj.t, traj.u[3, :])
    else
        # State is (A1, A2, B1, B2)
        A1_itp = make_interpolator_square(traj.t, traj.u[1, :])
        A2_itp = make_interpolator_square(traj.t, traj.u[2, :])
    end
    
    p = traj.source.params

    # Size the domain to the orbit extent plus margin on each side,
    # then shift by the co-moving time stamp so the front stays centred.
    t_min, t_max = extrema(traj.t)
    x_left  = t_min - margin + p.c0 * time_stamp
    x_right = t_max + margin + p.c0 * time_stamp
    dw      = (x_right - x_left) / 2.0

    dh  = dw / aspect_ratio
    ny::Int = vis_param.HEIGHT
    nx::Int = ny * aspect_ratio
    x = LinRange(x_left, x_right, nx)
    y = LinRange(-dh, dh, ny)
    z = signed_power_square(field_moving_square(A1_itp, A2_itp, p, x, y, time_stamp; rotated=rotated), gamma)
    zmax = maximum(abs.(z))
    zmax = iszero(zmax) ? 1.0 : zmax  # Avoid division by zero for trivial fixed point
    plt = heatmap(
        collect(x), collect(y), z;
        color        = vis_param.COLOR_MAP,
        aspect_ratio = :equal,
        axis         = false,
        colorbar     = color_bar,
        size         = (nx, ny),
        margin       = 0mm,
        clim=(-zmax, zmax),
    )
    filename = joinpath(run_info.folder, "fronts_square", "$(traj.name).png")
    mkpath(dirname(filename))
    savefig(plt, filename)
    println("  Saved → $filename")
end

function create_bulk_front_images_square(orbits::Vector{OrbitScoreSquare}, run_info::RunInfo, time_stamp::Float64=0.0, vis_param::visualization_params_square=VIS_PARAMS_FRONT_SQUARE, color_bar::Bool=true; rotated::Bool=false)
    println("\n" * "="^70)
    println("CREATING FRONT IMAGES FOR ORBITS (SQUARE)")
    println("="^70)
    for score in orbits
        create_front_image_square(score.trajectory, run_info, time_stamp, vis_param, color_bar; rotated=rotated)  # Adjust time offset for better front visualization
    end
end

# =============================================================================
# Data output functions

function write_trajectory_dat_square(traj::TRAJECTORY_SQUARE, run_info::RunInfo)
    mkpath(joinpath(run_info.folder, "trajectory_data_square"))
    filename = joinpath(run_info.folder, "trajectory_data_square", "$(traj.name).dat")
    
    p = traj.source.params
    if is_theta_zero(p.theta)
        # θ = 0 case: state is (A₁, B₁, A₂)
        open(filename, "w") do f
            println(f, "t A1 B1 A2")
            for i in 1:length(traj.t)
                println(f, "$(traj.t[i]) $(traj.u[1, i]) $(traj.u[2, i]) $(traj.u[3, i])")
            end
        end
    else
        # Standard case: state is (A₁, A₂, B₁, B₂)
        open(filename, "w") do f
            println(f, "t A1 A2 B1 B2")
            for i in 1:length(traj.t)
                println(f, "$(traj.t[i]) $(traj.u[1, i]) $(traj.u[2, i]) $(traj.u[3, i]) $(traj.u[4, i])")
            end
        end
    end
    println("Trajectory data (square) written to: $filename")
end

function plot_amplitudes_square(traj::TRAJECTORY_SQUARE, run_info::RunInfo)
    mkpath(joinpath(run_info.folder, "amplitude_plots_square"))
    filename = joinpath(run_info.folder, "amplitude_plots_square", "$(traj.name)_amplitudes.png")
    
    # Extract amplitudes based on state dimensionality
    if size(traj.u, 1) == 3
        # State is (A1, B1, A2)
        A1_data = traj.u[1, :]
        A2_data = traj.u[3, :]
    else
        # State is (A1, A2, B1, B2)
        A1_data = traj.u[1, :]
        A2_data = traj.u[2, :]
    end
    
    plt = plot(
        traj.t, A1_data, label="A1", linewidth=2,
        xlabel="Time", ylabel="Amplitude",
        title="Amplitudes for $(traj.name) (Square)",
        legend=:best, size=(1200, 800), dpi=150
    )
    plot!(plt, traj.t, A2_data, label="A2", linewidth=2)
    
    savefig(plt, filename)
    println("Amplitude plot (square) saved to: $filename")
end
