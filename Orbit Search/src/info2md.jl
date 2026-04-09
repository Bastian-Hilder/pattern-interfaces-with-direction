# Utility functions for creating timestamped output folders and writing run metadata
using Dates

struct RunInfo
    folder::String
end

function RunInfo(; label::String="")
    folder = new_run_folder(label)
    return RunInfo(folder)
end

# =============================================================================
# Output folder management

# Create a timestamped subdirectory under `base` and return its path.
# Optional `label` is appended after the timestamp.
function new_run_folder(label::String=""; base::String="results")
    stamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    folder_name = isempty(label) ? stamp : "$(stamp)_$(label)"
    folder_path = joinpath(base, folder_name)
    mkpath(folder_path)
    println("Created output folder: $folder_path")
    return folder_path
end

# =============================================================================
# Markdown writers

# Write info.md with a parameter table and fixed-point table.
function write_info_md(folder::String, p::Params, fps::Vector{FixedPoint})
    path = joinpath(folder, "info.md")
    open(path, "w") do io
        println(io, "# Run Info")
        println(io, "")
        println(io, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "")
        println(io, "## Parameters")
        println(io, "")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| K0        | $(p.K0) |")
        println(io, "| K2        | $(p.K2) |")
        println(io, "| beta2     | $(p.beta2) |")
        println(io, "| mu0       | $(p.mu0) |")
        println(io, "| c0        | $(p.c0) |")
        println(io, "| theta     | $(round(p.theta * 180 / π, digits=4))° |")
        println(io, "| T         | $(p.T) |")
        println(io, "")
        println(io, "## Fixed Points")
        println(io, "")
        println(io, "| Name | A1 | A2 | A3 | Energy | Unstable dim |")
        println(io, "|------|----|----|----|--------|--------------|")
        for fp in fps
            A1, A2, A3 = fp.coords
            println(io, "| $(fp.name) | $(round(A1, sigdigits=6)) | $(round(A2, sigdigits=6)) | $(round(A3, sigdigits=6)) | $(round(fp.energy, sigdigits=6)) | $(fp.dimension_unstable) |")
        end
    end
    println("Wrote $path")
end

# Build a markdown table string from a vector of OrbitScores.
function orbits_table_md(scores::Vector{OrbitScore})
    io = IOBuffer()
    println(io, "| Name | Source | Target FP | eps | Perturbation | Time Near | Min Distance |")
    println(io, "|------|--------|-----------|-----|--------------|-----------|--------------|")
    for s in scores
        traj = s.trajectory
        pert = "[" * join(round.(traj.perturbation, sigdigits=4), ", ") * "]"
        println(io, "| $(traj.name) | $(traj.source.name) | $(s.fp.name) | $(traj.eps) | $(pert) | $(round(s.time_near, digits=4)) | $(round(s.min_distance, sigdigits=4)) |")
    end
    return String(take!(io))
end

# Append a new section (## section_title) to an existing info.md.
function append_info_md(folder::String, section_title::String, content::String)
    path = joinpath(folder, "info.md")
    open(path, "a") do io
        println(io, "")
        println(io, "## $section_title")
        println(io, "")
        print(io, content)
    end
end

# =============================================================================
# Write comprehensive fixed point data to multiple files

function write_fixed_point_data(fp_factory::Function, fp_name::String, p::Params, mumin::Float64, mumax::Float64, step_size::Float64, run_info::RunInfo)
    step_size > 0 || throw(ArgumentError("step_size must be positive"))
    
    output_dir = run_info.folder
    
    # Initialize arrays to track all data
    mu_values = Float64[]
    dim_unstable_values = Int[]
    dim_stable_values = Int[]
    coords_A1 = Float64[]
    coords_A2 = Float64[]
    coords_A3 = Float64[]
    
    # Collect data by scanning over mu0
    for mu0 in mumin:step_size:mumax
        p_mu = Params(p.K0, p.K2, p.beta2, mu0, p.c0, p.theta, p.T)
        try
            fp = fp_factory(p_mu)
            push!(mu_values, mu0)
            push!(dim_unstable_values, fp.dimension_unstable)
            push!(dim_stable_values, fp.dimension_stable)
            push!(coords_A1, fp.coords[1])
            push!(coords_A2, fp.coords[2])
            push!(coords_A3, fp.coords[3])
        catch
            # Skip this mu0 if fixed point doesn't exist
        end
    end
    
    # Write consolidated data file
    open(joinpath(output_dir, "$(fp_name).dat"), "w") do f
        println(f, "mu0 A1 A2 dim_unstable")
        for i in 1:length(mu_values)
            println(f, "$(mu_values[i]) $(coords_A1[i]) $(coords_A2[i]) $(dim_unstable_values[i])")
        end
    end
    
    # Detect dimension changes (bifurcation points)
    dimension_changes = []
    
    for i in 2:length(mu_values)
        if dim_unstable_values[i] != dim_unstable_values[i-1]
            push!(dimension_changes, (mu_values[i], "unstable", dim_unstable_values[i-1], dim_unstable_values[i]))
        end
        if dim_stable_values[i] != dim_stable_values[i-1]
            push!(dimension_changes, (mu_values[i], "stable", dim_stable_values[i-1], dim_stable_values[i]))
        end
    end
    
    # Write markdown bifurcation report
    open(joinpath(output_dir, "$(fp_name)_bifurcations.md"), "w") do f
        println(f, "# Bifurcation Analysis: $fp_name")
        println(f, "")
        println(f, "Parameter scan: μ₀ ∈ [$mumin, $mumax] with step size $step_size")
        println(f, "")
        
        if isempty(dimension_changes)
            println(f, "No dimension changes detected in the scanned range.")
        else
            println(f, "## Dimension Changes")
            println(f, "")
            println(f, "| μ₀ (approx) | Subspace | Old Dimension | New Dimension |")
            println(f, "|-------------|----------|---------------|---------------|")
            for (mu, type, old_dim, new_dim) in dimension_changes
                println(f, "| $mu | $type | $old_dim | $new_dim |")
            end
        end
        
        println(f, "")
        println(f, "## Summary")
        println(f, "")
        if !isempty(mu_values)
            println(f, "- **Initial** (μ₀ = $(mu_values[1])):")
            println(f, "  - Unstable dimension: $(dim_unstable_values[1])")
            println(f, "  - Stable dimension: $(dim_stable_values[1])")
            println(f, "")
            println(f, "- **Final** (μ₀ = $(mu_values[end])):")
            println(f, "  - Unstable dimension: $(dim_unstable_values[end])")
            println(f, "  - Stable dimension: $(dim_stable_values[end])")
        else
            println(f, "No data collected (fixed point may not exist in this range).")
        end
    end
    
    println("Data written to $output_dir")
    println("  - Data file: $(fp_name).dat")
    println("  - Bifurcation report: $(fp_name)_bifurcations.md")
end

# =============================================================================
# Write comprehensive fixed point data for SQUARE LATTICE

function write_fixed_point_data_square(fp_factory::Function, fp_name::String, p::Params, mumin::Float64, mumax::Float64, step_size::Float64, run_info::RunInfo)
    step_size > 0 || throw(ArgumentError("step_size must be positive"))
    
    output_dir = run_info.folder
    
    # Initialize arrays to track all data
    mu_values = Float64[]
    dim_unstable_values = Int[]
    dim_stable_values = Int[]
    coords_A1 = Float64[]
    coords_A2 = Float64[]
    
    # Collect data by scanning over mu0
    for mu0 in mumin:step_size:mumax
        p_mu = Params(p.K0, p.K2, p.beta2, mu0, p.c0, p.theta, p.T)
        try
            fp = fp_factory(p_mu)
            push!(mu_values, mu0)
            push!(dim_unstable_values, fp.dimension_unstable)
            push!(dim_stable_values, fp.dimension_stable)
            push!(coords_A1, fp.coords[1])
            push!(coords_A2, fp.coords[2])
        catch
            # Skip this mu0 if fixed point doesn't exist
        end
    end
    
    # Write consolidated data file
    open(joinpath(output_dir, "$(fp_name).dat"), "w") do f
        println(f, "mu0 A1 A2 dim_unstable")
        for i in 1:length(mu_values)
            println(f, "$(mu_values[i]) $(coords_A1[i]) $(coords_A2[i]) $(dim_unstable_values[i])")
        end
    end
    
    # Detect dimension changes (bifurcation points)
    dimension_changes = []
    
    for i in 2:length(mu_values)
        if dim_unstable_values[i] != dim_unstable_values[i-1]
            push!(dimension_changes, (mu_values[i], "unstable", dim_unstable_values[i-1], dim_unstable_values[i]))
        end
        if dim_stable_values[i] != dim_stable_values[i-1]
            push!(dimension_changes, (mu_values[i], "stable", dim_stable_values[i-1], dim_stable_values[i]))
        end
    end
    
    # Write markdown bifurcation report
    open(joinpath(output_dir, "$(fp_name)_bifurcations.md"), "w") do f
        println(f, "# Bifurcation Analysis (Square): $fp_name")
        println(f, "")
        println(f, "Parameter scan: μ₀ ∈ [$mumin, $mumax] with step size $step_size")
        println(f, "")
        
        if isempty(dimension_changes)
            println(f, "No dimension changes detected in the scanned range.")
        else
            println(f, "## Dimension Changes")
            println(f, "")
            println(f, "| μ₀ (approx) | Subspace | Old Dimension | New Dimension |")
            println(f, "|-------------|----------|---------------|---------------|")
            for (mu, type, old_dim, new_dim) in dimension_changes
                println(f, "| $mu | $type | $old_dim | $new_dim |")
            end
        end
        
        println(f, "")
        println(f, "## Summary")
        println(f, "")
        if !isempty(mu_values)
            println(f, "- **Initial** (μ₀ = $(mu_values[1])):")
            println(f, "  - Unstable dimension: $(dim_unstable_values[1])")
            println(f, "  - Stable dimension: $(dim_stable_values[1])")
            println(f, "")
            println(f, "- **Final** (μ₀ = $(mu_values[end])):")
            println(f, "  - Unstable dimension: $(dim_unstable_values[end])")
            println(f, "  - Stable dimension: $(dim_stable_values[end])")
        else
            println(f, "No data collected (fixed point may not exist in this range).")
        end
    end
    
    println("Data written to $output_dir")
    println("  - Data file: $(fp_name).dat")
    println("  - Bifurcation report: $(fp_name)_bifurcations.md")
end
