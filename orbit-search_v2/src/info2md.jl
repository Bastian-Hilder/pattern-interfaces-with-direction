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
