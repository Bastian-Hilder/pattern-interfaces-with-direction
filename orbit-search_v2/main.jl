using LinearAlgebra

using Pkg
Pkg.activate(@__DIR__); Pkg.instantiate()

include("src/parameters.jl")
include("src/fixed-points.jl")
include("src/shooting.jl")
include("src/info2md.jl")
include("src/visuals.jl")

# =============================================================================
# turn functions on or of

const create_fixed_point_images::Bool = false
const shoot_from_unstable_directions::Bool = true
const coarse_search_for_interesting_orbits::Bool = false
const fine_search_for_interesting_orbits::Bool = false
const best_search_for_interesting_orbits::Bool = false
const do_sweep::Bool = false
const write_energy_data::Bool = false

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
            traj = shoot_to_most_unstable(source,1e-3)
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

function main()
    # Set parameters
    run_info = RunInfo()


    p = Params(K0=-3.0,K2=-5.0,beta2=0.5,theta=π/12)

    mumm = mixed_mode_bif_point(p)
    println("Mixed-mode bifurcation point (mu0): $mumm")
    
    # instantiate fixed points
    fps = instantiate_fixed_points(p)

    # Print summary of fixed points
    println("\n--- Fixed point analysis for current parameters ---")
    for fp in fps
        print_fixed_point(fp)
    end

    # Create output folder and write parameter/fixed-point info
    write_info_md(run_info.folder, p, fps)

    # Render wavefield images for each fixed point
    if create_fixed_point_images
        create_all_fixed_point_images(fps,run_info)
    end

    # Example: shoot from fixed points in most unstable direction

    if shoot_from_unstable_directions
        println("\n" * "="^70)
        println("SHOOTING FROM UNSTABLE DIRECTIONS")
        println("="^70)
        for fp in fps
            if fp.dimension_unstable >0
                traj = shoot_to_most_unstable(fp)
                if traj !== nothing
                    create_front_image(traj, run_info, -40/fp.params.c0)
                end
            end
        end
    end

    # Example Coarse search for interesting orbits from a source fixed point
    if coarse_search_for_interesting_orbits
        source = hexagon_fp(p)
        interesting_orbits = find_interesting_orbits_coarse(source, fps, 1e-3, 0.1, 100000, 10)
        append_info_md(run_info.folder, "Coarse Orbit Search", orbits_table_md(interesting_orbits))
        println("\n--- Interesting orbits from $(source.name) ---")
        for score in interesting_orbits
            create_front_image(score.trajectory, run_info, -40/source.params.c0)  # Adjust time offset for better front visualization
            println(score)
        end
    end

    if fine_search_for_interesting_orbits
        source = hexagon_fp(p)
        interesting_orbits = fine_search(source, fps, 1e-3, 0.1, 100000, 10)
        append_info_md(run_info.folder, "Fine Orbit Search", orbits_table_md(interesting_orbits))
        println("\n--- Fine interesting orbits from $(source.name) ---")
        for score in interesting_orbits
            create_front_image(score.trajectory, run_info, -40/source.params.c0)  # Adjust time offset for better front visualization
            println(score)
        end
    end

    if best_search_for_interesting_orbits
        source = hexagon_fp(p)
        interesting_orbits = find_best_orbit(source, fps, 1e-3, 0.1, 10000)
        append_info_md(run_info.folder, "Best Orbit Search", orbits_table_md(interesting_orbits))
        println("\n--- Best interesting orbits from $(source.name) ---")
        for score in interesting_orbits
            create_front_image(score.trajectory, run_info, -50/source.params.c0,VIS_PARAMS_FRONT_WIDE)  # Adjust time offset for better front visualization
            println(score)
        end
    end

    if do_sweep
        sweep()
    end

    if write_energy_data
        p = Params(K0=-0.3,K2=-0.6)

        xmax = 500.0

        dist = 0.5

        mufold = hexagon_fold_point(p)

        mumm = mixed_mode_bif_point(p)

        # Write roll wave
        write_energy(roll_wave_fp, p, 0.0, xmax, dist, joinpath(run_info.folder, "roll_wave_energy$(p.K0)_$(p.K2)_$(xmax).dat"))
        # Write hexagon
        write_energy(hexagon_fp, p, 0.0, xmax, dist, joinpath(run_info.folder, "hexagon_energy$(p.K0)_$(p.K2)_$(xmax).dat"))
        # Write down-hexagon
        write_energy(down_hexagon_fp, p, 0.0, xmax, dist, joinpath(run_info.folder, "down_hexagon_energy$(p.K0)_$(p.K2)_$(xmax).dat"))
        # Write mixed-mode
        write_energy(mixed_mode_fp, p, mumm, xmax, dist, joinpath(run_info.folder, "mixed_mode_energy$(p.K0)_$(p.K2)_$(xmax).dat"))
    end
    
end


# -----------------------------------------------------------------------------
# Parameter sweep for fine_search
function sweep()
    K0K2_pairs = [(-3.0, -6.0), (-6.0, -3.0)]
    mu0_values = [0.5, 1.0, 2.0]
    theta_values = [0.0, π/18, π/9, π/6 - 0.01]  # 0°, 10°, 20°, ~29.9°

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
