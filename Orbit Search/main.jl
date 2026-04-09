using LinearAlgebra
 
using Pkg
Pkg.activate(@__DIR__); Pkg.instantiate()
 
include("src/parameters.jl")
include("src/fixed-points.jl")
include("src/fixed-points-square.jl")
include("src/shooting.jl")
include("src/shooting-square.jl")
include("src/info2md.jl")
include("src/visuals.jl")
include("src/visuals-square.jl")

 
# =============================================================================
# Toggle functions on or off
 
const create_fixed_point_images::Bool         = false
const create_fixed_point_images_square::Bool  = false
const shoot_from_unstable_directions::Bool    = false
const shoot_from_unstable_directions_square::Bool = false
const coarse_search_for_interesting_orbits::Bool = false
const fine_search_for_interesting_orbits::Bool   = false
const best_search_for_interesting_orbits::Bool   = false
const do_sweep::Bool                          = false
const write_energy_data::Bool                 = false
const write_energy_data_square::Bool          = false
const write_fixed_point_data_toggle::Bool     = false
const write_fixed_point_data_square_toggle::Bool = true
const shoot_from_lowest_energy_toggle::Bool   = false
const shoot_from_lowest_energy_square_toggle::Bool = false
const shoot_from_all_fps_toggle::Bool         = false
const shoot_from_all_fps_square_toggle::Bool  = false
 
# =============================================================================
# Shooting-based search
 
function run_fine_search(p::Params)
    run_info = RunInfo()
 
    fps = instantiate_fixed_points(p)
 
    println("\n--- Fixed point analysis for current parameters ---")
    for fp in fps
        print_fixed_point(fp)
    end
 
    write_info_md(run_info.folder, p, fps)
 
    for source in fps
        if source.dimension_unstable == 1
            traj = shoot_to_most_unstable(source, 1e-3)
            if traj !== nothing
                create_front_image(traj, run_info, -40/source.params.c0)
            end
        elseif source.dimension_unstable > 1
            interesting_orbits = fine_search(source, fps, 1e-3, 0.2, 20000, 10)
            append_info_md(run_info.folder, "Fine Orbit Search", orbits_table_md(interesting_orbits))
            println("\n--- Fine search results from $(source.name) ---")
            for score in interesting_orbits
                create_front_image(score.trajectory, run_info, -40/source.params.c0)
                println(score)
            end
        end
    end
end
 
# =============================================================================
 
function main()
    run_info = RunInfo()
 
    p = Params()
 
    mufold = hexagon_fold_point(p)
    println("Hexagon fold bifurcation point (mu0): $mufold")
 
    mumm = mixed_mode_bif_point(p)
    println("Mixed-mode bifurcation point (mu0): $mumm")
 
    fps = instantiate_fixed_points(p)
 
    println("\n--- Fixed point analysis for current parameters ---")
    for fp in fps
        print_fixed_point(fp)
    end
 
    write_info_md(run_info.folder, p, fps)
 
    if create_fixed_point_images
        create_fp_images(p, mufold, 7.0, 0.1, run_info, color_bar=false)
    end

    if create_fixed_point_images_square
        create_fp_images_square(p, 0.0, 7.0, 0.1, run_info, color_bar=true)
    end
 
    if shoot_from_unstable_directions
        println("\n" * "="^70)
        println("SHOOTING FROM UNSTABLE DIRECTIONS")
        println("="^70)
        for fp in fps
            if fp.dimension_unstable > 0
                traj = shoot_to_most_unstable(fp)
                if traj !== nothing
                    create_front_image(traj, run_info, -40/fp.params.c0; rotated=true)
                end
            end
        end
    end

    if shoot_from_unstable_directions_square
        println("\n" * "="^70)
        println("SHOOTING FROM UNSTABLE DIRECTIONS (SQUARE)")
        println("="^70)
        fps_square = instantiate_fixed_points_square(p)
        for fp in fps_square
            if fp.dimension_unstable > 0
                traj = shoot_to_most_unstable_square(fp)
                if traj !== nothing
                    create_front_image_square(traj, run_info)
                end
            end
        end
    end
 
    if coarse_search_for_interesting_orbits
        source = hexagon_fp(p)
        interesting_orbits = find_interesting_orbits_coarse(source, fps, 1e-3, 0.1, 100000, 10)
        append_info_md(run_info.folder, "Coarse Orbit Search", orbits_table_md(interesting_orbits))
        println("\n--- Interesting orbits from $(source.name) ---")
        for score in interesting_orbits
            create_front_image(score.trajectory, run_info, -40/source.params.c0)
            println(score)
        end
    end
 
    if fine_search_for_interesting_orbits
        source = hexagon_fp(p)
        interesting_orbits = fine_search(source, fps, 1e-3, 0.1, 100000, 10)
        append_info_md(run_info.folder, "Fine Orbit Search", orbits_table_md(interesting_orbits))
        println("\n--- Fine interesting orbits from $(source.name) ---")
        for score in interesting_orbits
            create_front_image(score.trajectory, run_info, -40/source.params.c0)
            println(score)
        end
    end
 
    if best_search_for_interesting_orbits
        source = hexagon_fp(p)
        interesting_orbits = find_best_orbit(source, fps, 1e-3, 0.1, 10000)
        append_info_md(run_info.folder, "Best Orbit Search", orbits_table_md(interesting_orbits))
        println("\n--- Best interesting orbits from $(source.name) ---")
        for score in interesting_orbits
            create_front_image(score.trajectory, run_info, -50/source.params.c0, VIS_PARAMS_FRONT_WIDE)
            println(score)
        end
    end
 
    if do_sweep
        sweep()
    end

    if shoot_from_lowest_energy_toggle
        shoot_from_lowest_energy()
    end

    if shoot_from_all_fps_toggle
        shoot_from_all_fps()
    end

    if shoot_from_lowest_energy_square_toggle
        shoot_from_lowest_energy_square()
    end

    if shoot_from_all_fps_square_toggle
        shoot_from_all_fps_square()
    end
 
    if write_energy_data
        p = Params(K0=-1.2, K2=-0.6)
        xmax = 1.1
        dist = 0.001
        mufold = hexagon_fold_point(p)
        mumm   = mixed_mode_bif_point(p)
        println("\n--- Writing energy data for parameters: K0=$(p.K0), K2=$(p.K2) ---")

        write_energy(roll_wave_fp,    p, 0.0,  xmax, dist, joinpath(run_info.folder, "roll_wave_energy$(p.K0)_$(p.K2)_$(xmax).dat"))
        write_energy(hexagon_fp,      p, 0.0,  xmax, dist, joinpath(run_info.folder, "hexagon_energy$(p.K0)_$(p.K2)_$(xmax).dat"))
        write_energy(down_hexagon_fp, p, 0.0,  xmax, dist, joinpath(run_info.folder, "down_hexagon_energy$(p.K0)_$(p.K2)_$(xmax).dat"))
        write_energy(mixed_mode_fp,   p, mumm, xmax, dist, joinpath(run_info.folder, "mixed_mode_energy$(p.K0)_$(p.K2)_$(xmax).dat"))

        println("--- Energy data written to files in $(run_info.folder) ---")
    end

    if write_fixed_point_data_toggle
        write_fp_data()
    end

    if write_energy_data_square
        write_energy_data_square_func()
    end

    if write_fixed_point_data_square_toggle
        write_fp_data_square()
    end
end

function shoot_from_lowest_energy()
    K0K2_pairs  = [(-3.0, -6.0), (-6.0, -3.0)]
    theta_values = [0.0, π/18, π/9, π/6 - 0.01]
    mumin = 0.1
    mumax = 2.0
    step_size = 0.1

    for (K0, K2) in K0K2_pairs
        for theta in theta_values
            # One results folder per (K0, K2, theta) group
            run_info = RunInfo(label="shoot_from_lowest_energy_K0_$(K0)_K2_$(K2)_theta_$(round(theta*180/π, digits=2))")

            for mu0 in mumin:step_size:mumax
                p_mu = Params(K0 = K0, K2 = K2, beta2 = 1.0, mu0 = mu0, c0 = 8.0, theta = theta, T = 100.0)
                fps = instantiate_fixed_points(p_mu)

                # Exclude trivial fixed point — we want the lowest-energy patterned state
                non_trivial_fps = filter(fp -> fp.name != "trivial", fps)
                if isempty(non_trivial_fps)
                    println("\n--- No non-trivial fixed points for K0=$K0, K2=$K2, mu0=$(round(mu0, digits=3)), theta=$(round(theta*180/π, digits=2))° — skipping ---")
                    continue
                end

                sorted_fps = sort(non_trivial_fps, by=fp -> fp.energy)
                lowest_energy_fp = first(sorted_fps)

                println("\n--- Shooting from lowest-energy non-trivial fp for K0=$K0, K2=$K2, mu0=$(round(mu0, digits=3)), theta=$(round(theta*180/π, digits=2))° ---")
                println("Fixed point: $(lowest_energy_fp.name) | energy: $(lowest_energy_fp.energy)")

                traj = shoot_to_most_unstable(lowest_energy_fp)
                if traj !== nothing
                    # Prefix the trajectory name with mu0 so fronts are distinguishable
                    # within the shared folder
                    traj_named = TRAJECTORY(
                        "mu0_$(round(mu0, digits=3))_$(lowest_energy_fp.name)",
                        traj.source, traj.eps, traj.perturbation,
                        traj.exceeded_threshold, traj.t, traj.u
                    )
                    create_front_image_full(traj_named, run_info)
                end
            end
        end
    end
end

function shoot_from_all_fps()
    K0K2_pairs   = [(-3.0, -6.0), (-6.0, -3.0)]
    theta_values = [0.0, π/18, π/9, π/6 - 0.01]
    mumin     = 1.0
    mumax     = 1.0
    step_size = 0.5

    for (K0, K2) in K0K2_pairs
        c_crit = critical_speed(Params(K0=K0, K2=K2, beta2=1.0, mu0=1.0, c0=8.0, theta=0.0, T=100.0))
        println("\n--- Critical speed for K0=$K0, K2=$K2: $c_crit ---")
        c0_values = [c_crit - 2.0, c_crit, c_crit + 2.0]
        for theta in theta_values
            for c0 in c0_values
                # One results folder per (K0, K2, theta, c0) group
                run_info = RunInfo(label="shoot_all_fps_K0_$(K0)_K2_$(K2)_theta_$(round(theta*180/π, digits=2))_c0_$(round(c0, digits=1))")

                for mu0 in mumin:step_size:mumax
                    p_mu = Params(K0 = K0, K2 = K2, beta2 = 1.0, mu0 = mu0, c0 = c0, theta = theta, T = 100.0)
                    fps  = instantiate_fixed_points(p_mu)

                    non_trivial_fps = filter(fp -> fp.name != "trivial", fps)
                    if isempty(non_trivial_fps)
                        println("\n--- No non-trivial fps for K0=$K0, K2=$K2, c0=$c0, mu0=$(round(mu0, digits=3)), theta=$(round(theta*180/π, digits=2))° — skipping ---")
                        continue
                    end

                    for fp in non_trivial_fps
                        println("\n--- Shooting from $(fp.name) for K0=$K0, K2=$K2, c0=$c0, mu0=$(round(mu0, digits=3)), theta=$(round(theta*180/π, digits=2))° ---")
                        traj = shoot_to_most_unstable(fp)
                        if traj !== nothing
                            traj_named = TRAJECTORY(
                                "mu0_$(round(mu0, digits=3))_c0_$(round(c0, digits=1))_$(fp.name)",
                                traj.source, traj.eps, traj.perturbation,
                                traj.exceeded_threshold, traj.t, traj.u
                            )
                            create_front_image_shifted(traj_named, run_info, 0.0; color_bar=false, gamma=0.4)
                            write_trajectory_dat(traj_named, run_info)
                            plot_amplitudes(traj_named, run_info)
                        end
                    end
                end
            end
        end
    end
end

# =============================================================================
# Write fixed point data across parameter ranges

function write_fp_data()
    run_info = RunInfo(label="fixed_point_data")
    
    p = Params(K0=-3.0, K2=-6.0)
    mumin = 0.0
    mumax = 20.0
    step_size = 0.01
    
    mufold = hexagon_fold_point(p)
    mumm   = mixed_mode_bif_point(p)
    
    println("\n--- Writing fixed point data for K0=$(p.K0), K2=$(p.K2) ---")
    println("Hexagon fold point: $mufold")
    println("Mixed-mode bif point: $mumm")
    println("Scan range: μ₀ ∈ [$mumin, $mumax] with step $step_size")
    
    # Write data for each fixed point type
    write_fixed_point_data(trivial_fp,      "trivial",      p, mumin, mumax, step_size, run_info)
    write_fixed_point_data(roll_wave_fp,    "roll_wave",    p, mumin, mumax, step_size, run_info)
    write_fixed_point_data(hexagon_fp,      "hexagon",      p, mufold, mumax, step_size, run_info)
    write_fixed_point_data(down_hexagon_fp, "down_hexagon", p, mufold, mumax, step_size, run_info)
    write_fixed_point_data(mixed_mode_fp,   "mixed_mode",   p, mumm, mumax, step_size, run_info)
    
    println("\n--- All fixed point data written to $(run_info.folder) ---")
end

# =============================================================================
# Write energy data for SQUARE LATTICE

function write_energy_data_square_func()
    run_info = RunInfo(label="energy_data_square")
    
    p = Params(K0=-1.2, K2=-0.6)
    mumin = 0.0
    mumax = 1.1
    step_size = 0.001
    
    println("\n--- Writing energy data (square) for K0=$(p.K0), K2=$(p.K2) ---")
    
    write_energy_square(trivial_fp_square, p, mumin, mumax, step_size, joinpath(run_info.folder, "trivial_energy_square.dat"))
    write_energy_square(roll_wave_fp_square, p, mumin, mumax, step_size, joinpath(run_info.folder, "roll_wave_energy_square.dat"))
    write_energy_square(square_fp, p, mumin, mumax, step_size, joinpath(run_info.folder, "square_energy_square.dat"))
    
    println("--- Energy data (square) written to $(run_info.folder) ---")
end

# =============================================================================
# Write fixed point data for SQUARE LATTICE

function write_fp_data_square()
    run_info = RunInfo(label="fixed_point_data_square")
    
    p = Params(K0=-0.6, K2=-1.2, theta=pi/4)
    mumin = 0.0
    mumax = 20.0
    step_size = 0.01
    
    println("\n--- Writing fixed point data (square) for K0=$(p.K0), K2=$(p.K2) ---")
    println("Scan range: μ₀ ∈ [$mumin, $mumax] with step $step_size")
    
    # Write data for each fixed point type
    write_fixed_point_data_square(trivial_fp_square, "trivial_square", p, mumin, mumax, step_size, run_info)
    write_fixed_point_data_square(roll_wave_fp_square, "roll_wave_square", p, mumin, mumax, step_size, run_info)
    write_fixed_point_data_square(square_fp, "square", p, mumin, mumax, step_size, run_info)
    
    println("\n--- All fixed point data (square) written to $(run_info.folder) ---")
end

# =============================================================================
# Shoot from lowest energy - SQUARE LATTICE

function shoot_from_lowest_energy_square()
    K0K2_pairs  = [(-3.0, -6.0), (-6.0, -3.0)]
    theta_values = [0.0, π/4, π/3]
    mumin = 0.1
    mumax = 2.0
    step_size = 0.1

    for (K0, K2) in K0K2_pairs
        for theta in theta_values
            run_info = RunInfo(label="shoot_from_lowest_energy_square_K0_$(K0)_K2_$(K2)_theta_$(round(theta*180/π, digits=2))")

            for mu0 in mumin:step_size:mumax
                p_mu = Params(K0 = K0, K2 = K2, beta2 = 1.0, mu0 = mu0, c0 = 8.0, theta = theta, T = 100.0)
                fps = instantiate_fixed_points_square(p_mu)

                non_trivial_fps = filter(fp -> fp.name != "trivial", fps)
                if isempty(non_trivial_fps)
                    println("\n--- No non-trivial fixed points (square) for K0=$K0, K2=$K2, mu0=$(round(mu0, digits=3)), theta=$(round(theta*180/π, digits=2))° — skipping ---")
                    continue
                end

                sorted_fps = sort(non_trivial_fps, by=fp -> fp.energy)
                lowest_energy_fp = first(sorted_fps)

                println("\n--- Shooting from lowest-energy non-trivial fp (square) for K0=$K0, K2=$K2, mu0=$(round(mu0, digits=3)), theta=$(round(theta*180/π, digits=2))° ---")
                println("Fixed point: $(lowest_energy_fp.name) | energy: $(lowest_energy_fp.energy)")

                traj = shoot_to_most_unstable_square(lowest_energy_fp)
                if traj !== nothing
                    traj_named = TRAJECTORY_SQUARE(
                        "mu0_$(round(mu0, digits=3))_$(lowest_energy_fp.name)",
                        traj.source, traj.eps, traj.perturbation,
                        traj.exceeded_threshold, traj.t, traj.u
                    )
                    create_front_image_full_square(traj_named, run_info)
                end
            end
        end
    end
end

# =============================================================================
# Shoot from all fps - SQUARE LATTICE

function shoot_from_all_fps_square()
    K0K2_pairs   = [(-3.0, -6.0), (-6.0, -3.0)]
    theta_values = [0.0, π/18, π/9, π/6 - 0.01]
    mumin     = 1.0
    mumax     = 1.0
    step_size = 0.5

    for (K0, K2) in K0K2_pairs
        c_crit = critical_speed_square(Params(K0=K0, K2=K2, beta2=1.0, mu0=1.0, c0=8.0, theta=0.0, T=100.0))
        println("\n--- Critical speed (square) for K0=$K0, K2=$K2: $c_crit ---")
        c0_values = [c_crit - 2.0, c_crit, c_crit + 2.0]
        for theta in theta_values
            for c0 in c0_values
                run_info = RunInfo(label="shoot_all_fps_square_K0_$(K0)_K2_$(K2)_theta_$(round(theta*180/π, digits=2))_c0_$(round(c0, digits=1))")

                for mu0 in mumin:step_size:mumax
                    p_mu = Params(K0 = K0, K2 = K2, beta2 = 1.0, mu0 = mu0, c0 = c0, theta = theta, T = 100.0)
                    fps  = instantiate_fixed_points_square(p_mu)

                    non_trivial_fps = filter(fp -> fp.name != "trivial", fps)
                    if isempty(non_trivial_fps)
                        println("\n--- No non-trivial fps (square) for K0=$K0, K2=$K2, c0=$c0, mu0=$(round(mu0, digits=3)), theta=$(round(theta*180/π, digits=2))° — skipping ---")
                        continue
                    end

                    for fp in non_trivial_fps
                        println("\n--- Shooting from $(fp.name) (square) for K0=$K0, K2=$K2, c0=$c0, mu0=$(round(mu0, digits=3)), theta=$(round(theta*180/π, digits=2))° ---")
                        traj = shoot_to_most_unstable_square(fp)
                        if traj !== nothing
                            traj_named = TRAJECTORY_SQUARE(
                                "mu0_$(round(mu0, digits=3))_c0_$(round(c0, digits=1))_$(fp.name)",
                                traj.source, traj.eps, traj.perturbation,
                                traj.exceeded_threshold, traj.t, traj.u
                            )
                            create_front_image_shifted_square(traj_named, run_info, 0.0; color_bar=false, gamma=0.4)
                            write_trajectory_dat_square(traj_named, run_info)
                            plot_amplitudes_square(traj_named, run_info)
                        end
                    end
                end
            end
        end
    end
end
 
# =============================================================================
# Parameter sweep for fine_search
 
function sweep()
    K0K2_pairs  = [(-3.0, -6.0), (-6.0, -3.0)]
    mu0_values  = [0.5, 1.0, 2.0]
    theta_values = [0.0, π/18, π/9, π/6 - 0.01]
 
    for (K0, K2) in K0K2_pairs
        for mu0 in mu0_values
            for theta in theta_values
                println("\n" * "="^70)
                println("SWEEP: K0=$K0, K2=$K2, mu0=$mu0, theta=$(round(theta*180/π, digits=2))°")
                println("="^70)
                p = Params(K0=K0, K2=K2, mu0=mu0, theta=theta)
                run_fine_search(p)
            end
        end
    end
end
 
main()