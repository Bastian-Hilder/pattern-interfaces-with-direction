# Create all visualisations for fixed points and heteroclinic orbits
using Plots, Interpolations

HIGH_QUALITY = true  # Set to false for quick rendering during development

DEFAULT_HEIGHT      = HIGH_QUALITY ? 2160 : 1080
DEFAULT_WIDTH       = HIGH_QUALITY ? 3840 : 1920
DEFAULT_DOMAIN_SIZE = 4π   # half-width in each spatial direction
DEFAULT_DOMAIN_SIZE_FRONT = 20π  # Larger domain for front screenshots
DOMAIN_SIZE_FRONT_WIDE = 40π  # Even wider domain for front screenshots with large aspect ratio

struct visualization_params
    HEIGHT::Int
    WIDTH::Int
    COLOR_MAP::Symbol
    domain_size::Float64
end

# =============================================================================
# Visualisation parameters
const VIS_PARAMS = visualization_params(DEFAULT_HEIGHT, DEFAULT_WIDTH, :jet, DEFAULT_DOMAIN_SIZE)
const VIS_PARAMS_FRONT = visualization_params(DEFAULT_HEIGHT, DEFAULT_WIDTH, :jet, DEFAULT_DOMAIN_SIZE_FRONT)
const VIS_PARAMS_FRONT_WIDE = visualization_params(DEFAULT_HEIGHT, DEFAULT_WIDTH, :jet, DOMAIN_SIZE_FRONT_WIDE)

# =============================================================================
# Wavefield

# Compute the real wavefield value at point (x, y):
#   u(r) = 2 * [A1 cos(k1·r) + A2 cos(k2·r) + A3 cos(k3·r)]
function wavefield_value(A1::Real, A2::Real, A3::Real, x::Real, y::Real)
    k1 = polar(0.0)
    k2 = polar(2π / 3)
    k3 = polar(-2π / 3)
    r  = [x, y]
    return 2.0 * (A1 * cos(dot(k1, r)) + A2 * cos(dot(k2, r)) + A3 * cos(dot(k3, r)))
end

# =============================================================================
# Create images of all fixed points for the current parameter set

function create_all_fixed_point_images(fps::Vector{FixedPoint},run_info::RunInfo, vis_param::visualization_params=VIS_PARAMS;color_bar::Bool=true)
    println("\n" * "="^70)
    println("CREATING FIXED-POINT IMAGES")
    println("="^70)
    mkpath(joinpath(run_info.folder, "fixed_points"))
    for fp in fps
        create_fixed_point_image(fp; filename=joinpath(run_info.folder, "fixed_points", "$(fp.name)_$(fp.params.K0)_$(fp.params.K2)_$(fp.params.mu0).png"), vis_param=vis_param, color_bar)
    end
end


function create_fp_images(p::Params, mumin::Float64, mumax::Float64, step_size::Float64, run_info::RunInfo, vis_param::visualization_params=VIS_PARAMS;color_bar::Bool=true)
    step_size > 0 || throw(ArgumentError("step_size must be positive"))
    for mu0 in mumin:step_size:mumax
        p_mu = Params(p.K0, p.K2, p.beta2, mu0, p.c0, p.theta, p.T)
        try
            fps = instantiate_fixed_points(p_mu)
            create_all_fixed_point_images(fps, run_info, vis_param, color_bar=color_bar)
         catch error
            @warn "Failed to create fixed point images for mu0 = $mu0: $error"
        end
    end
end 



function create_fixed_point_image(fp::FixedPoint; filename::String, vis_param::visualization_params=VIS_PARAMS,color_bar::Bool=true)
    println("\nCreating image for fixed point: $(fp.name)")
    A1, A2, A3 = fp.coords
    d  = vis_param.domain_size
    nx = vis_param.HEIGHT # Use HEIGHT for both dimensions to ensure square aspect ratio
    ny = vis_param.HEIGHT
    x = LinRange(-d, d, nx)
    y = LinRange(-d, d, ny)
    z = [wavefield_value(A1, A2, A3, Float64(xi), Float64(yi)) for yi in y, xi in x]
    zmax = maximum(abs.(z))
    zmax = iszero(zmax) ? 1.0 : zmax  # Avoid division by zero for trivial fixed point
    plt = heatmap(
        collect(x), collect(y), z;
        color        = vis_param.COLOR_MAP,
        aspect_ratio = :equal,
        axis         = false,
        colorbar     = color_bar,
        size         = (nx, ny),
        clim=(-zmax, zmax),
    )
    savefig(plt, filename)
    println("  Saved → $filename")
end


# =============================================================================
# Create static images for fronts

function linear_interpolation(t::AbstractVector, A::AbstractVector)
    left_val  = A[1]    # source fixed point value
    right_val = A[end]  # destination fixed point value (wherever it converged to)
    t_min, t_max = extrema(t)
    itp = LinearInterpolation(t, A, extrapolation_bc=Flat())
    return xi -> xi < t_min ? left_val : xi > t_max ? right_val : itp(xi)
end

function field_moving(A1_itp, A2_itp, A3_itp, p::Params, x, y, t;rotated::Bool=false)
    k1, k2, k3, _, _, _ = kappas(p.theta;rotated=rotated)
    d = rotated ? polar(0) : polar(p.theta)

    X = repeat(x',      length(y), 1)
    Y = repeat(y,  1,   length(x))

    # Time coordinate in the co-moving frame
    t_sample = (d[1] .* X .+ d[2] .* Y) .- p.c0 .* t
    F = 2 * (A1_itp.(t_sample) .* cos.(k1[1] .* X .+ k1[2] .* Y) +
         A2_itp.(t_sample) .* cos.(k2[1] .* X .+ k2[2] .* Y) +
         A3_itp.(t_sample) .* cos.(k3[1] .* X .+ k3[2] .* Y))

    return F
end

function create_front_image_full(traj::TRAJECTORY, run_info::RunInfo, 
                             vis_param::visualization_params=VIS_PARAMS_FRONT,
                             color_bar::Bool=true; rotated::Bool=false)
    println("\nCreating front image for trajectory: $(traj.name)")
 
    A1_itp = linear_interpolation(traj.t, traj.u[1, :])
    A2_itp = linear_interpolation(traj.t, traj.u[2, :])
    A3_itp = linear_interpolation(traj.t, traj.u[3, :])
 
    p = traj.source.params
 
    t_min, t_max = extrema(traj.t)
    L_orbit = (t_max - t_min) / 2.0          # half the orbit length
    centre  = (t_max + t_min) / 2.0  # centre in x
    dw      = L_orbit + 2.0     # +2 margin
 
    aspect_ratio::Integer = 5
    dh  = dw / aspect_ratio
    ny  = vis_param.HEIGHT
    nx  = ny * aspect_ratio
 
    x = LinRange(centre - dw, centre + dw, nx)
    y = LinRange(-dh, dh, ny)
 
    z    = field_moving(A1_itp, A2_itp, A3_itp, p, x, y, 0; rotated=rotated)
    zmax = maximum(abs.(z))
    zmax = iszero(zmax) ? 1.0 : zmax
 
    plt = heatmap(
        collect(x), collect(y), z;
        color        = vis_param.COLOR_MAP,
        aspect_ratio = :equal,
        axis         = false,
        colorbar     = color_bar,
        size         = (nx, ny),
        clim         = (-zmax, zmax),
    )
 
    filename = joinpath(run_info.folder, "fronts", "$(traj.name).png")
    mkpath(dirname(filename))
    savefig(plt, filename)
    println("  Saved → $filename")
end

function create_front_image(traj::TRAJECTORY,run_info::RunInfo,time_stamp::Float64 = 0.0, vis_param::visualization_params=VIS_PARAMS_FRONT,color_bar::Bool=true;rotated::Bool=false)
    println("\nCreating front image for trajectory: $(traj.name)")
    A1_itp = linear_interpolation(traj.t, traj.u[1, :])
    A2_itp = linear_interpolation(traj.t, traj.u[2, :])
    A3_itp = linear_interpolation(traj.t, traj.u[3, :])    
    p = traj.source.params
    aspect_ratio::Integer = 5
    dh = vis_param.domain_size / aspect_ratio
    dw = vis_param.domain_size
    ny::Int = vis_param.HEIGHT
    nx::Int = ny * aspect_ratio  # Adjust width to maintain aspect ratio
    x = LinRange(-dw, dw, nx)
    y = LinRange(-dh, dh, ny)
    z = field_moving(A1_itp, A2_itp, A3_itp, p, x, y, time_stamp; rotated=rotated)
    zmax = maximum(abs.(z))
    zmax = iszero(zmax) ? 1.0 : zmax  # Avoid division by zero for trivial fixed point
    plt = heatmap(
        collect(x), collect(y), z;
        color        = vis_param.COLOR_MAP,
        aspect_ratio = :equal,
        axis         = false,
        colorbar     = color_bar,
        size         = (nx, ny),
        clim=(-zmax, zmax),
    )
    filename = joinpath(run_info.folder, "fronts", "$(traj.name).png")
    mkpath(dirname(filename))
    savefig(plt, filename)
    println("  Saved → $filename")
end

function create_bulk_front_images(orbits::Vector{OrbitScore}, run_info::RunInfo,time_stamp::Float64 = 0.0, vis_param::visualization_params=VIS_PARAMS_FRONT,color_bar::Bool=true;rotated::Bool=false)
    println("\n" * "="^70)
    println("CREATING FRONT IMAGES FOR ORBITS")
    println("="^70)
    for score in orbits
        create_front_image(score.trajectory, run_info, time_stamp, vis_param, color_bar; rotated=rotated)  # Adjust time offset for better front visualization
    end
end