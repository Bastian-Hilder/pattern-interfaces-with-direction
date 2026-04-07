# bvp_cont.jl
#
# Heteroclinic orbit continuation using:
#   - BoundaryValueDiffEq (MIRK4)  for the BVP solve at each step
#   - BifurcationKit (PALC)        for pseudo-arclength continuation in μ₀
#
# Architecture
# ────────────
# BifurcationKit needs   F(x, p) = 0   where x ∈ ℝ^(6N) is the orbit
# flattened onto a fixed uniform grid and p is a named parameter struct.
#
# We define F(x, p) = x_solved(x, p) - x :
#   given a candidate orbit x, solve the BVP using x as the initial guess,
#   resample the BVP solution back onto the same grid, and return the
#   difference.  A genuine solution satisfies F = 0 (fixed-point residual).
#
# BifurcationKit differentiates F via finite differences to build its
# Jacobian.  The internal BVP solve is treated as a black box.
#
# Boundary conditions
# ───────────────────
# Left  BC: Vs_source' * (u(-L) - p⁻) = 0   → u(-L) ∈ p⁻ + W^u(p⁻)
# Right BC: Vu_target' * (u(+L) - p⁺) = 0   → u(+L) ∈ p⁺ + W^s(p⁺)
# Phase:    u(-L)[4] = B₁(-L) = 0            → kills ξ-translation freedom
#
# Dimension count: n_left + n_right + 1 = 6
# Requires: dim W^u(source) + dim W^s(target) = 7  (generic transversality)
 
using BoundaryValueDiffEq
using BifurcationKit
using LinearAlgebra
using Interpolations
using Printf
 
# ─────────────────────────────────────────────────────────────────────────────
# Grid size — fixed for the whole continuation
# ─────────────────────────────────────────────────────────────────────────────
 
const N_GRID = 200
 
# ─────────────────────────────────────────────────────────────────────────────
# 1.  BUILD INITIAL GUESS
# ─────────────────────────────────────────────────────────────────────────────
 
function build_bvp_guess(traj::TRAJECTORY,
                          source::FixedPoint,
                          target::FixedPoint,
                          L::Union{Float64,Nothing}=nothing)
    t_raw  = traj.t
    u_raw  = traj.u
    L_auto = max(maximum(t_raw) / 2.0, 30.0)
    L_use  = isnothing(L) ? L_auto : L
 
    t_mid  = (t_raw[1] + t_raw[end]) / 2.0
    tshift = t_raw .- t_mid
 
    t_grid  = collect(LinRange(-L_use, L_use, N_GRID))
    u_guess = zeros(6, N_GRID)
    for row in 1:6
        itp = LinearInterpolation(tshift, u_raw[row, :], extrapolation_bc=Flat())
        u_guess[row, :] = itp.(t_grid)
    end
    return t_grid, u_guess, L_use
end
 
# ─────────────────────────────────────────────────────────────────────────────
# 2.  ODE RHS
# ─────────────────────────────────────────────────────────────────────────────
 
function ode_rhs!(du, u, p::Params, t)
    du .= F(u, p)
end
 
# ─────────────────────────────────────────────────────────────────────────────
# 3.  BOUNDARY CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────
 
function make_bc(source::FixedPoint, target::FixedPoint)
    Vs_source   = source.stable_subspace    # 6 × (6 - k⁻)
    Vu_target   = target.unstable_subspace  # 6 × k⁺
    source_full = vcat(source.coords, zeros(3))
    target_full = vcat(target.coords, zeros(3))
    n_left      = size(Vs_source, 2)
    n_right     = size(Vu_target, 2)
    n_bc        = n_left + n_right   # must equal 6; SparseConnectivityTracer allocates residual of size 6

    if n_bc != 6
        @warn "make_bc: n_bc=$n_bc ≠ 6 (dim W^u(source)=$(source.dimension_unstable), dim W^s(target)=$(target.dimension_stable)). BVP may be ill-posed."
    end

    function bc!(residual, sol, p, t)
        u_start = sol[1]
        u_end   = sol[end]
        residual[1:n_left]                = Vs_source' * (u_start - source_full)
        residual[n_left+1:n_left+n_right] = Vu_target' * (u_end   - target_full)
    end

    return bc!, n_bc
end
 
# ─────────────────────────────────────────────────────────────────────────────
# 4.  BVP SOLVER  (BoundaryValueDiffEq)
# ─────────────────────────────────────────────────────────────────────────────
 
function solve_bvp(source::FixedPoint, target::FixedPoint, p::Params;
                   t_guess::AbstractVector,
                   u_guess::Matrix{Float64},
                   alg=MIRK4(),
                   abstol=1e-6)
    bc!, _ = make_bc(source, target)
 
    function guess_fn(t)
        idx = clamp(searchsortedfirst(t_guess, t), 1, length(t_guess))
        return u_guess[:, idx]
    end
 
    t_span = (t_guess[1], t_guess[end])
    prob   = BVProblem(ode_rhs!, bc!, guess_fn, t_span, p)
    return solve(prob, alg; abstol=abstol, dt=(t_span[2]-t_span[1])/N_GRID)
end
 
# ─────────────────────────────────────────────────────────────────────────────
# 5.  BIFURCATIONKIT PARAMETER STRUCT
# ─────────────────────────────────────────────────────────────────────────────
# BK varies one field of this struct (mu0) via @optic _.mu0.
# All other fields are fixed context passed through the continuation.
 
struct BKParams
    K0::Float64
    K2::Float64
    beta2::Float64
    mu0::Float64       # ← continuation parameter
    c0::Float64
    theta::Float64
    T::Float64
end
 
function to_params(bkp::BKParams)
    return Params(bkp.K0, bkp.K2, bkp.beta2, bkp.mu0, bkp.c0, bkp.theta, bkp.T)
end
 
# ─────────────────────────────────────────────────────────────────────────────
# 6.  BIFURCATIONKIT RESIDUAL  F(x, bkp) = x_solved - x
# ─────────────────────────────────────────────────────────────────────────────
 
function make_bk_residual(source_factory::Function,
                           target_factory::Function,
                           t_grid::Vector{Float64},
                           abstol::Float64)
    N = length(t_grid)
 
    function F_bk(x::AbstractVector, bkp::BKParams)
        p      = to_params(bkp)
        source = source_factory(p)
        target = target_factory(p)
 
        u_guess = reshape(real.(x), 6, N)
 
        sol = solve_bvp(source, target, p;
                        t_guess=t_grid,
                        u_guess=u_guess,
                        abstol=abstol)
 
        # Resample solution onto fixed grid and return fixed-point residual
        u_solved = hcat([sol(ti) for ti in t_grid]...)
        return vec(u_solved) - real.(x)
    end
 
    return F_bk
end
 
# ─────────────────────────────────────────────────────────────────────────────
# 7.  RECORD CALLBACK  (called by BK at each accepted step)
# ─────────────────────────────────────────────────────────────────────────────
 
function make_record_fn(t_grid::Vector{Float64},
                         source_factory::Function,
                         target_factory::Function,
                         run_info::Union{RunInfo,Nothing})
    N = length(t_grid)
 
    function record(x, bkp; k...)
        u_mat  = reshape(collect(Float64, x), 6, N)
        mid    = u_mat[:, N÷2]
        p      = to_params(bkp)
        target = target_factory(p)
        dist   = norm(u_mat[1:3, end] - target.coords)
 
        if run_info !== nothing
            try
                source  = source_factory(p)
                udir, _ = most_unstable_direction(source)
                traj    = TRAJECTORY("bvp-mu$(round(bkp.mu0, sigdigits=5))",
                                      source, 0.0, udir, false,
                                      t_grid, u_mat)
                mkpath(joinpath(run_info.folder, "fronts"))
                create_front_image_full(traj, run_info)
            catch e
                @warn "Front image failed at μ₀=$(bkp.mu0): $e"
            end
        end
 
        return (A1_mid     = mid[1],
                A2_mid     = mid[2],
                A3_mid     = mid[3],
                B1_mid     = mid[4],
                dist_target = dist)
    end
 
    return record
end
 
# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
 
"""
    find_and_continue_heteroclinic(p0, source_name, target_name;
                                   mu0_start, mu0_end, ...)
 
Find and continue a heteroclinic orbit using PALC.
 
Returns a BifurcationKit `ContResult`.  Each point on the branch has:
  pt.param           → μ₀ value
  pt.record.A1_mid   → A₁ at midpoint ξ=0
  pt.record.dist_target → endpoint distance to target fixed point
 
Usage:
    p0 = Params(K0=-3.0, K2=-6.0, beta2=1.0, mu0=0.5, c0=8.0, theta=0.0, T=100.0)
    br = find_and_continue_heteroclinic(p0, "hexagon", "roll_wave";
             mu0_start=0.5, mu0_end=2.5, run_info=RunInfo())
"""
function find_and_continue_heteroclinic(p0::Params,
                                         source_name::String,
                                         target_name::String;
                                         mu0_start::Float64,
                                         mu0_end::Float64,
                                         ds::Float64          = 0.05,
                                         ds_max::Float64      = 0.1,
                                         ds_min::Float64      = 1e-4,
                                         max_steps::Int       = 200,
                                         abstol::Float64      = 1e-6,
                                         run_info::Union{RunInfo,Nothing} = nothing,
                                         verbose::Bool        = true)
 
    # ── Fixed-point factories ─────────────────────────────────────────────
    function source_factory(p::Params)
        fps = instantiate_fixed_points(p)
        idx = findfirst(fp -> fp.name == source_name, fps)
        idx === nothing && error("Fixed point '$source_name' not found at μ₀=$(p.mu0)")
        return fps[idx]
    end
 
    function target_factory(p::Params)
        fps = instantiate_fixed_points(p)
        idx = findfirst(fp -> fp.name == target_name, fps)
        idx === nothing && error("Fixed point '$target_name' not found at μ₀=$(p.mu0)")
        return fps[idx]
    end
 
    # ── Seed orbit via shooting ───────────────────────────────────────────
    p_start = Params(p0.K0, p0.K2, p0.beta2, mu0_start, p0.c0, p0.theta, p0.T)
    source0 = source_factory(p_start)
    target0 = target_factory(p_start)
 
    verbose && println("\n── Shooting seed orbit at μ₀ = $mu0_start ──")
    seed = shoot_to_most_unstable(source0, 1e-3)
    seed === nothing && error("Shooting failed — cannot seed BVP.")
 
    t_grid, u_guess, _ = build_bvp_guess(seed, source0, target0)
 
    # ── Initial BVP solve ─────────────────────────────────────────────────
    verbose && println("── Initial BVP solve at μ₀ = $mu0_start ──")
    sol0  = solve_bvp(source0, target0, p_start;
                      t_guess=t_grid, u_guess=u_guess, abstol=abstol)
    dist0 = norm(sol0(t_grid[end])[1:3] - target0.coords)
    verbose && @printf("   endpoint distance to target: %.2e\n", dist0)
    dist0 > 0.1 && @warn "Initial endpoint distance = $(round(dist0, sigdigits=2)) — solution may not be heteroclinic"
 
    # Flatten onto fixed grid → initial state vector for BK
    x0 = vec(hcat([sol0(ti) for ti in t_grid]...))
 
    # ── BifurcationKit setup ──────────────────────────────────────────────
    bkp0   = BKParams(p0.K0, p0.K2, p0.beta2, mu0_start, p0.c0, p0.theta, p0.T)
    F_bk   = make_bk_residual(source_factory, target_factory, t_grid, abstol)
    rec_fn = make_record_fn(t_grid, source_factory, target_factory, run_info)

    # Finite-difference Jacobian: BifurcationKit's default uses ForwardDiff, which
    # pushes Dual numbers through F_bk into the BVP solver — that fails because
    # the BVP solver only accepts Float64. We supply an explicit FD Jacobian instead.
    function J_bk(x, bkp)
        n  = length(x)
        Fx = F_bk(x, bkp)
        J  = zeros(n, n)
        h  = sqrt(abstol)
        for j in 1:n
            xp      = copy(x); xp[j] += h
            J[:, j] = (F_bk(xp, bkp) - Fx) / h
        end
        return J
    end

    prob = BifurcationProblem(F_bk, x0, bkp0, (@optic _.mu0);
                               J                    = J_bk,
                               record_from_solution = rec_fn)
 
    opts = ContinuationPar(
        p_min              = min(mu0_start, mu0_end),
        p_max              = max(mu0_start, mu0_end),
        ds                 = sign(mu0_end - mu0_start) * abs(ds),
        dsmax              = ds_max,
        dsmin              = ds_min,
        max_steps          = max_steps,
        detect_bifurcation = 0,
        newton_options     = NewtonPar(tol=abstol, max_iterations=20,
                                       verbose=false),
    )
 
    verbose && println("\n── PALC continuation  μ₀: $mu0_start → $mu0_end ──\n")
 
    br = continuation(prob, PALC(tangent=Secant()), opts;
                      verbosity = verbose ? 1 : 0)
 
    verbose && println("\n── Done: $(length(br)) points on branch ──")
 
    if run_info !== nothing
        write_branch_data(br, run_info)
    end
 
    print_branch_summary(br)
    return br
end
 
# ─────────────────────────────────────────────────────────────────────────────
# 9.  OUTPUT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
 
function write_branch_data(br, run_info::RunInfo)
    fname = joinpath(run_info.folder, "heteroclinic_branch.dat")
    open(fname, "w") do f
        println(f, "mu0 A1_mid A2_mid A3_mid dist_target")
        for pt in br.branch
            r = pt.record
            @printf(f, "%.8f  %.8f  %.8f  %.8f  %.6e\n",
                    pt.param, r.A1_mid, r.A2_mid, r.A3_mid, r.dist_target)
        end
    end
    println("Wrote $fname")
end
 
function print_branch_summary(br)
    println("\n── Heteroclinic branch summary ──────────────────────────────────")
    println("   μ₀          A₁(0)      A₂(0)      A₃(0)      dist(p⁺)")
    println("  " * "─"^62)
    for pt in br.branch
        r = pt.record
        @printf("  %10.5f   %9.5f  %9.5f  %9.5f  %.3e\n",
                pt.param, r.A1_mid, r.A2_mid, r.A3_mid, r.dist_target)
    end
end
 
"""
    get_orbit(br, step, t_grid) -> (t_grid, u_matrix)
 
Extract the orbit at a given step index as a (t_grid, 6×N) pair for
post-processing or visualisation.
"""
function get_orbit(br, step::Int, t_grid::Vector{Float64})
    x = br.sol[step].x
    return t_grid, reshape(collect(Float64, x), 6, length(t_grid))
end