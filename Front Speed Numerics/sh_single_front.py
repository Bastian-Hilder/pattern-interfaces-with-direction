from mpi4py import MPI
import dolfinx
from dolfinx import default_real_type, log, plot, mesh, fem
from dolfinx.fem import functionspace, Function
from dolfinx.fem.petsc import NonlinearProblem

from basix.ufl import element, mixed_element

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import os

import ufl

from dolfinx.io import XDMFFile
import csv
from datetime import datetime

# Generate timestamp for output files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# static simulation parameters

t = 0
T = 100
# For fourth-order parabolic: dt ~ O(Δx²) for accuracy with implicit methods
N = T * 50  # 1000 time steps, dt ≈ 0.02
dt = T / N
theta = 0.5
hexagon_scaling = 1 / np.sqrt(3)

# Swift-Hohenberg parameters
eps  = 0.3
mu0  = 1.0
mu   = eps**2 * mu0
beta = eps
beta2 = 1

K0 = -3 
K2 = -6

# Hexagonal pattern amplitude
discriminant = beta2**2 - 4*mu0*(K0 + 2*K2)
Ahex_plus = (-beta2 + np.sqrt(discriminant)) / (2*(K0 + 2*K2))
Ahex_minus = (-beta2 - np.sqrt(discriminant)) / (2*(K0 + 2*K2))
Ahex = Ahex_minus  # Choose the physical solution

localisation_factor = 3  # controls transition width at x=0
front_angle = 0.0  # angle of front direction in radians (0 = along x-axis, π/2 = along y-axis)

# Front tracking parameters
front_threshold = 0.05
front_data_file = f"single_front_position_{timestamp}.csv"
rolling_profile_file = f"single_front_rolling_profile_{timestamp}.csv"

# Visualization parameters
plot_dir = f"single_front_plots_{timestamp}"
if MPI.COMM_WORLD.rank == 0:
    os.makedirs(plot_dir, exist_ok=True)

file_name = f"demo_ch/single_front_SH_{timestamp}.xdmf"

# Generate mesh
mesh_comm = MPI.COMM_WORLD

left_x = -4 * np.pi
domain_size_x = 20*2*np.pi  # reduced domain in x-direction
domain_size_y = 3*2*np.pi * hexagon_scaling   # extended in y-direction
# Doubled resolution: ~63 points per wavelength (λ = 2π)
unit_grid_points = 15

mesh_x = int(domain_size_x * unit_grid_points)
mesh_y = int(2 * domain_size_y * unit_grid_points)

domain = mesh.create_rectangle(
    mesh_comm, 
    [[left_x, -domain_size_y], [left_x + domain_size_x, domain_size_y]], 
    [mesh_x, mesh_y], 
    mesh.CellType.triangle
)

print(f"Start single front simulation on domain [{left_x}, {left_x + domain_size_x}]x[{-domain_size_y}, {domain_size_y}]")
print(f"Mesh resolution: {mesh_x} x {mesh_y} grid points")

# Define Function Space
P1 = element("Lagrange", domain.basix_cell(), 1, dtype=default_real_type)
V = functionspace(domain, mixed_element([P1, P1]))

U  = Function(V)
U0 = Function(V)

u,  w  = ufl.split(U)
u0, w0 = ufl.split(U0)

v, q = ufl.TestFunctions(V)

# Initial condition: pattern for x < 0, cut-off for x > 0

def initial_condition_single_front(x):
    """
    Hexagonal pattern with front at angle theta.
    Pattern is active on one side, decays on the other.
    """
    period = 1
    k1 = period * x[0]
    k2 = period * (-0.5*x[0] + 0.5*np.sqrt(3)*x[1])
    k3 = period * (-0.5*x[0] - 0.5*np.sqrt(3)*x[1])
    hex_pattern = 2*Ahex*(np.cos(k1) + np.cos(k2) + np.cos(k3))
    
    # Rotate coordinates by front_angle
    # Front normal direction
    x_rotated = x[0] * np.cos(front_angle) + x[1] * np.sin(front_angle)

    # Smooth cutoff: 1 on one side, decays to 0 on the other
    cutoff = 0.5 * (1.0 - np.tanh(x_rotated / localisation_factor))
    
    return eps * hex_pattern * cutoff

U.sub(0).interpolate(initial_condition_single_front)
U.x.scatter_forward()

# Initialise w = (1+Δ)u via L² projection
V0, _ = V.sub(0).collapse()
u_init = U.sub(0).collapse()

w_init_problem = fem.petsc.LinearProblem(
    ufl.inner(ufl.TrialFunction(V0), ufl.TestFunction(V0)) * ufl.dx,
    ufl.inner(u_init, ufl.TestFunction(V0)) * ufl.dx
    - ufl.inner(ufl.grad(u_init), ufl.grad(ufl.TestFunction(V0))) * ufl.dx,
    petsc_options_prefix='w_init'
)
w_init = w_init_problem.solve()

_, dofs_w = V.sub(1).collapse()
U.x.array[dofs_w] = w_init.x.array
U.x.scatter_forward()

U0.x.array[:] = U.x.array

# Mid-point values
u_mid = (1.0 - theta) * u0 + theta * u
w_mid = (1.0 - theta) * w0 + theta * w

# Weak formulation
F0 = (
    ufl.inner((u - u0) / dt, v) * ufl.dx
    + ufl.inner(w_mid, v) * ufl.dx
    - ufl.inner(ufl.grad(w_mid), ufl.grad(v)) * ufl.dx
    - mu * ufl.inner(u_mid, v) * ufl.dx
    + beta * ufl.inner(ufl.dot(ufl.grad(u_mid), ufl.grad(u_mid)), v) * ufl.dx
    + ufl.inner(u_mid**3, v) * ufl.dx
)

F1 = (
    ufl.inner(w, q) * ufl.dx
    - ufl.inner(u, q) * ufl.dx
    + ufl.inner(ufl.grad(u), ufl.grad(q)) * ufl.dx
)

F = F0 + F1

# Set PETSc options
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": np.sqrt(np.finfo(default_real_type).eps) * 1e-2,
}

problem = NonlinearProblem(F, U, petsc_options_prefix='sh_newton', petsc_options=petsc_options)

file = XDMFFile(MPI.COMM_WORLD, file_name, "w")
file.write_mesh(domain)

# Function to compute single front position
def compute_single_front_position(u_func, threshold=0.05, angle=0.0):
    """
    Track front position by computing L²-norm in bins spanning one period (2π).
    Returns the outermost position where the norm exceeds threshold.
    """
    u_collapsed = u_func.collapse()
    coords = u_collapsed.function_space.tabulate_dof_coordinates()
    u_vals = u_collapsed.x.array

    # Project onto front-normal direction
    x_rot = coords[:, 0] * np.cos(angle) + coords[:, 1] * np.sin(angle)

    x_min, x_max = np.min(x_rot), np.max(x_rot)
    
    # Bin width = one full period (wavelength = 2π)
    period = 2.0 * np.pi
    num_bins = int(np.ceil((x_max - x_min) / period))
    bin_edges = np.linspace(x_min, x_max, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Compute L²-norm in each period-wide bin
    l2_norm = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (x_rot >= bin_edges[i]) & (x_rot < bin_edges[i+1])
        if np.sum(mask) > 3:
            l2_norm[i] = np.sqrt(np.mean(u_vals[mask]**2))

    # Find outermost bin where L²-norm > threshold/sqrt(2)
    # (factor accounts for oscillatory pattern: RMS of A*cos = A/sqrt(2))
    env_threshold = threshold / np.sqrt(2)
    above = l2_norm > env_threshold

    if not np.any(above):
        return x_min

    last_idx = np.where(above)[0][-1]
    return bin_centers[last_idx]

def compute_rolling_l2_profile(u_func, angle=0.0, strip_width=2.0*np.pi, step_size=None):
    """
    Compute L²-norm in rolling strips along the front-normal direction.
    
    Args:
        u_func: The function to analyze
        angle: Front angle in radians
        strip_width: Width of each strip (default: 2π, one period)
        step_size: Step size for rolling window (default: strip_width/100)
    
    Returns:
        a_values: Array of strip starting positions
        l2_norms: Array of L²-norms for each strip
    """
    u_collapsed = u_func.collapse()
    coords = u_collapsed.function_space.tabulate_dof_coordinates()
    u_vals = u_collapsed.x.array

    # Project onto front-normal direction
    x_rot = coords[:, 0] * np.cos(angle) + coords[:, 1] * np.sin(angle)

    x_min, x_max = np.min(x_rot), np.max(x_rot)
    
    if step_size is None:
        step_size = strip_width / 100.0  # 100 samples per period for smooth tracking
    
    # Generate strip starting positions
    # We can go from x_min to (x_max - strip_width)
    max_a = x_max - strip_width
    if max_a <= x_min:
        # Domain too small
        return np.array([x_min]), np.array([0.0])
    
    a_values = np.arange(x_min, max_a + step_size/2, step_size)
    l2_norms = np.zeros(len(a_values))
    
    for i, a in enumerate(a_values):
        # Select points in strip [a, a + strip_width]
        mask = (x_rot >= a) & (x_rot < a + strip_width)
        n_points = np.sum(mask)
        
        if n_points > 3:
            # Compute RMS (root mean square)
            l2_norms[i] = np.sqrt(np.mean(u_vals[mask]**2))
        else:
            l2_norms[i] = 0.0
    
    return a_values, l2_norms

def compute_rolling_front_position(u_func, threshold=0.05, angle=0.0, strip_width=2.0*np.pi):
    """
    Find outermost position where rolling L²-norm exceeds threshold.
    
    Returns the largest 'a' such that L²-norm in [a, a + strip_width] > threshold.
    """
    a_values, l2_norms = compute_rolling_l2_profile(u_func, angle=angle, strip_width=strip_width)
    
    # Adjust threshold for oscillatory pattern
    env_threshold = threshold / np.sqrt(2)
    
    # Find largest 'a' where norm exceeds threshold
    above = l2_norms > env_threshold
    
    if not np.any(above):
        return a_values[0]  # Return leftmost position
    
    last_idx = np.where(above)[0][-1]
    return a_values[last_idx]

# Function to save heatmap
def save_single_front_heatmap(u_func, time_val, left_x, domain_size_x, domain_size_y, plot_directory):
    """Save heatmap of single front solution."""
    if MPI.COMM_WORLD.rank != 0:
        return
    
    u_collapsed = u_func.collapse()
    u_space = u_collapsed.function_space
    coords = u_space.tabulate_dof_coordinates()
    u_vals = u_collapsed.x.array
    
    # Create regular grid
    nx_plot = 500
    ny_plot = 200
    
    x_min = left_x
    x_max = left_x + domain_size_x
    x_grid = np.linspace(x_min, x_max, nx_plot)
    y_grid = np.linspace(-domain_size_y, domain_size_y, ny_plot)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Interpolate
    from scipy.interpolate import griddata
    Z = griddata(coords[:, :2], u_vals, (X, Y), method='linear', fill_value=0.0)
    
    zmax = np.max(np.abs(u_vals))
    if zmax == 0:
        zmax = 1.0
    
    # Calculate aspect ratio to maintain correct proportions
    aspect_ratio = (2 * domain_size_y) / domain_size_x
    fig_width = 12
    fig_height = fig_width * aspect_ratio
    
    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    im = ax.imshow(Z, extent=[x_min, x_max, -domain_size_y, domain_size_y],
                   origin='lower', cmap='jet', vmin=-zmax, vmax=zmax, aspect='equal')
    
    # Remove axes, labels, ticks
    ax.set_axis_off()
    
    filename = os.path.join(plot_directory, f'single_front_t{int(time_val):03d}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    print(f"  → Saved heatmap: {filename}")

# Initialize front tracking file
if MPI.COMM_WORLD.rank == 0:
    with open(front_data_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'x_front', 'x_front_rolling', 'max_amplitude'])
    
    # Initialize rolling profile file (will append profiles at integer times)
    with open(rolling_profile_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'a', 'l2_norm'])

# Time-stepping loop
t = 0.0
last_integer_time = -1

# Save initial condition
save_single_front_heatmap(U.sub(0), t, left_x, domain_size_x, domain_size_y, plot_dir)

while t < T:
    t += dt
    problem.solve()
    u_values = U.sub(0).x.array
    num_iterations = problem.solver.getIterationNumber()
    max_amplitude = np.max(np.abs(u_values))
    
    # Compute front position (original method)
    x_front = compute_single_front_position(U.sub(0), threshold=front_threshold, angle=front_angle)
    
    # Compute rolling front position
    x_front_rolling = compute_rolling_front_position(U.sub(0), threshold=front_threshold, angle=front_angle)
    
    print(f"Step {int(round(t / dt))}: iter: {num_iterations}, max|u| = {max_amplitude:.4e}, "
          f"front_pos = {x_front:.3f}, front_rolling = {x_front_rolling:.3f}")
    
    # Save front position data
    if MPI.COMM_WORLD.rank == 0:
        with open(front_data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([t, x_front, x_front_rolling, max_amplitude])
    
    # Save heatmap at integer time steps
    current_integer_time = int(np.floor(t))
    if current_integer_time > last_integer_time and current_integer_time > 0:
        save_single_front_heatmap(U.sub(0), float(current_integer_time), left_x, domain_size_x, domain_size_y, plot_dir)
        
        # Save rolling L²-profile at multiples of 10
        if current_integer_time % 10 == 0 and MPI.COMM_WORLD.rank == 0:
            a_values, l2_norms = compute_rolling_l2_profile(U.sub(0), angle=front_angle)
            with open(rolling_profile_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for a, norm in zip(a_values, l2_norms):
                    writer.writerow([current_integer_time, a, norm])
            print(f"  → Saved rolling L²-profile at t = {current_integer_time}")
        
        last_integer_time = current_integer_time
    
    U0.x.array[:] = U.x.array
    file.write_function(U.sub(0), t)

file.close()

print(f"\nSingle front position data saved to: {front_data_file}")
print(f"Rolling L²-profile data saved to: {rolling_profile_file}")
print(f"Heatmap plots saved to: {plot_dir}/")
