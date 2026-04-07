using LinearAlgebra
 
using Pkg
Pkg.activate(@__DIR__); Pkg.instantiate()
 
include("src/parameters.jl")
include("src/fixed-points.jl")
include("src/shooting.jl")
include("src/info2md.jl")
include("src/visuals.jl")
include("src/bvp_cont.jl")
include("src/bvp.jl")
 
# =============================================================================
# Toggle functions on or off
 
const create_fixed_point_images::Bool         = false
const shoot_from_unstable_directions::Bool    = false
const coarse_search_for_interesting_orbits::Bool = false
const fine_search_for_interesting_orbits::Bool   = false
const best_search_for_interesting_orbits::Bool   = false
const find_connecting_orbits_bvp::Bool        = true
const do_sweep::Bool                          = false
const write_energy_data::Bool                 = false
 
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
# BVP + BifurcationKit continuation
 
function run_heteroclinic_continuation()
    run_info = RunInfo(label="heteroclinic_continuation")
 
    p0 = Params(K0=-3.0, K2=-6.0, beta2=1.0, mu0=0.5, c0=8.0, theta=0.0, T=100.0)
 
    println("\nHexagon fold point:   $(hexagon_fold_point(p0))")
    println("Mixed-mode bif point: $(mixed_mode_bif_point(p0))")
 
    fps = instantiate_fixed_points(p0)
    write_info_md(run_info.folder, p0, fps)
 
    br = find_and_continue_heteroclinic(p0, "hexagon", "roll_wave";
             mu0_start = 0.5,
             mu0_end   = 2.5,
             ds        = 0.05,
             ds_max    = 0.1,
             ds_min    = 1e-4,
             max_steps = 200,
             abstol    = 1e-6,
             run_info  = run_info,
             verbose   = true)
 
    return br
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
 
    if find_connecting_orbits_bvp
        println("\n" * "="^70)
        println("BVP HETEROCLINIC CONTINUATION")
        println("="^70)
        run_heteroclinic_continuation()
    end
 
    if do_sweep
        sweep()
    end
 
    if write_energy_data
        p = Params(K0=-0.3, K2=-0.6)
        xmax = 500.0
        dist = 0.5
        mufold = hexagon_fold_point(p)
        mumm   = mixed_mode_bif_point(p)
        write_energy(roll_wave_fp,    p, 0.0,  xmax, dist, joinpath(run_info.folder, "roll_wave_energy$(p.K0)_$(p.K2)_$(xmax).dat"))
        write_energy(hexagon_fp,      p, 0.0,  xmax, dist, joinpath(run_info.folder, "hexagon_energy$(p.K0)_$(p.K2)_$(xmax).dat"))
        write_energy(down_hexagon_fp, p, 0.0,  xmax, dist, joinpath(run_info.folder, "down_hexagon_energy$(p.K0)_$(p.K2)_$(xmax).dat"))
        write_energy(mixed_mode_fp,   p, mumm, xmax, dist, joinpath(run_info.folder, "mixed_mode_energy$(p.K0)_$(p.K2)_$(xmax).dat"))
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