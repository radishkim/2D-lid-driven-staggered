# 2D Lid-Driven Cavity Flow Solver

A Python CFD solver for two-dimensional incompressible lid-driven cavity
flow.

The implementation includes:

- A staggered finite-difference grid
- A projection method for pressure-velocity coupling
- Second-order Adams-Bashforth treatment of convection
- ADI-style TDMA treatment of viscous terms
- DCT, SOR, multigrid, and BiCGSTAB pressure Poisson solvers
- Periodic checkpoints and safe `Ctrl+C` interruption
- Automatic `.npz` output and figure generation

## Quick Start

### 1. Install the dependencies

Run the following command from the project root:

```powershell
python -m pip install -r requirements.txt
```

Required packages:

```text
numpy
scipy
matplotlib
pytest
```

### 2. Select the Reynolds number

Edit `re` in [`src/lid_driven/config.py`](src/lid_driven/config.py):

```python
re: float = 100.0
```

For example, to run at `Re=500`:

```python
re: float = 500.0
```

### 3. Select the Poisson solver

Edit the following value in
[`scripts/run_cfd.py`](scripts/run_cfd.py):

```python
POISSON_SOLVER = "bicgstab"
```

Available values:

```python
"dct"
"sor"
"multigrid"
"bicgstab"
```

### 4. Run the simulation

```powershell
python scripts/run_cfd.py
```

With `Re=100` and `POISSON_SOLVER="bicgstab"`, the output is:

```text
results/data/solution_Re100_bicgstab.npz
```

### 5. Generate figures

```powershell
python scripts/plot_results.py
```

The figures are saved in:

```text
results/figures/Re100_bicgstab/
```

## Command-Line Solver Selection

The Poisson solver can be selected temporarily without editing
`POISSON_SOLVER`:

```powershell
python scripts/run_cfd.py dct
python scripts/run_cfd.py sor
python scripts/run_cfd.py multigrid
python scripts/run_cfd.py bicgstab
```

Use the same solver name when generating figures:

```powershell
python scripts/plot_results.py dct
python scripts/plot_results.py sor
python scripts/plot_results.py multigrid
python scripts/plot_results.py bicgstab
```

The Reynolds number is always read from `SimulationConfig.re`.

---

## Current Default Configuration

[`src/lid_driven/config.py`](src/lid_driven/config.py):

```python
nx: int = 130
ny: int = 130

re: float = 100.0
dt: float = 0.005
max_timestep: int = 1_000_000
checkpoint_interval: int = 1000

velocity_tolerance: float = 1.0e-5
u_lid: float = 1.0
```

[`scripts/run_cfd.py`](scripts/run_cfd.py):

```python
POISSON_SOLVER = "bicgstab"
```

> The solver selected by `run_cfd.py` overrides the default
> `poisson_solver` value in `config.py`.

The grid spacing is computed automatically:

```text
dx = (xmax - x0) / (nx - 2)
dy = (ymax - y0) / (ny - 2)
```

The terms `nx-2` and `ny-2` represent the numbers of interior pressure
cells after excluding the two ghost layers.

At higher Reynolds numbers, the default time step and grid resolution
may not be sufficient for stability or accuracy. Changing `re`
automatically changes the numerical coefficients and output paths, but
the suitability of `dt`, grid resolution, and convergence tolerances
must be evaluated separately.

---

## Overall Simulation Workflow

```text
scripts/run_cfd.py
    |
    +-- Create SimulationConfig
    +-- Generate plotting coordinates x and y
    +-- Determine the result-file path
    |
    +-- run_simulation()
            |
            +-- Initialize staggered-grid fields
            +-- Apply velocity boundary conditions
            |
            +-- Time-step loop
                    1. Compute the u- and v-momentum RHS
                    2. Compute viscous corrections du and dv with TDMA
                    3. Form intermediate velocities uhat and vhat
                    4. Compute the divergence of the intermediate field
                    5. Solve the pressure-correction Poisson equation
                    6. Project velocity onto a divergence-free field
                    7. Compute pressure
                    8. Compute the velocity convergence error
                    9. Save a checkpoint or stop after convergence
            |
            +-- Return the result dictionary
    |
    +-- Save the result as an .npz file
```

---

## Staggered-Grid Layout

The arrays are initialized in
[`src/lid_driven/grid.py`](src/lid_driven/grid.py).

| Variable | Shape | Location and purpose |
|---|---:|---|
| `p` | `(nx, ny)` | Cell-center pressure including ghost cells |
| `phi` | `(nx, ny)` | Pressure-correction field |
| `u` | `(nx - 1, ny)` | x-velocity on vertical cell faces |
| `v` | `(nx, ny - 1)` | y-velocity on horizontal cell faces |
| `up`, `vp` | Same as `u`, `v` | Velocity from the previous time step |
| `du`, `dv` | Same as `u`, `v` | Velocity corrections computed by TDMA |
| `uhat`, `vhat` | Same as `u`, `v` | Intermediate velocity before projection |

Storing pressure and velocity at different locations reduces
pressure-velocity decoupling and suppresses checkerboard pressure modes.

---

## Boundary Conditions

Velocity boundary conditions are applied by `apply_velocity_bc()` in
[`src/lid_driven/boundary.py`](src/lid_driven/boundary.py).

### Horizontal velocity `u`

- Left wall: `u = 0`
- Right wall: `u = 0`
- Bottom wall: antisymmetric ghost-cell condition
- Top wall: moving-lid condition

The top ghost-cell value is:

```text
u_ghost = 2 * u_lid - u_inside
```

Therefore, averaging the interior and ghost values at the wall gives
exactly `u_lid`.

### Vertical velocity `v`

- Bottom wall: `v = 0`
- Top wall: `v = 0`
- Left and right walls: antisymmetric ghost-cell conditions

The boundary conditions are applied during initialization, after the
intermediate-velocity calculation, and after projection.

---

## Momentum Equation

Momentum-equation operators are implemented in
[`src/lid_driven/operators.py`](src/lid_driven/operators.py).

### Convective terms

`compute_u_rhs()` computes:

```text
d(u^2)/dx + d(uv)/dy
```

`compute_v_rhs()` computes:

```text
d(uv)/dx + d(v^2)/dy
```

The face-centered values are interpolated from neighboring velocities.
The current and previous convective terms are combined using the
second-order Adams-Bashforth method:

```text
N^(n+1/2) = 1.5 * N^n - 0.5 * N^(n-1)
```

### Viscous terms

The velocity Laplacian is approximated using second-order central
differences:

```text
L(u) =
    (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx^2
  + (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dy^2
```

The momentum RHS has the form:

```text
rhs = -dt * convection + (dt / Re) * diffusion
```

---

## TDMA and Directional Splitting

`solve_u_diffusion_tdma()` and `solve_v_diffusion_tdma()` compute the
viscous velocity corrections `du` and `dv`.

The coefficients are:

```text
beta_x = dt / (2 * Re * dx^2)
beta_y = dt / (2 * Re * dy^2)
```

The `u` correction is computed using:

```text
x-direction TDMA
    -> y-direction TDMA
```

The `v` correction is computed using:

```text
y-direction TDMA
    -> x-direction TDMA
```

The tridiagonal systems are solved with the Thomas algorithm implemented
in [`src/lid_driven/linalg.py`](src/lid_driven/linalg.py).

The intermediate velocity is:

```text
uhat = u + du
vhat = v + dv
```

---

## Projection Method

The intermediate velocity is generally not divergence-free. A
pressure-correction Poisson equation is therefore solved before
updating the velocity.

### Poisson RHS

`compute_divergence()` in
[`src/lid_driven/diagnostics.py`](src/lid_driven/diagnostics.py)
computes:

```text
f = div(uhat) / dt
```

The discrete form is:

```text
f[i,j] =
    (
        (uhat[i,j] - uhat[i-1,j]) / dx
      + (vhat[i,j] - vhat[i,j-1]) / dy
    ) / dt
```

The pressure-correction equation is:

```text
Laplacian(phi) = f
```

### Velocity correction

`project_velocity()` subtracts the pressure-correction gradient from
the intermediate velocity:

```text
u^(n+1) = uhat - dt * d(phi)/dx
v^(n+1) = vhat - dt * d(phi)/dy
```

Velocity boundary conditions are applied again after projection.

### Pressure calculation

`update_pressure()` computes the pressure field from `phi` and a
Laplacian-like viscous correction.

The current implementation does not use the simple cumulative update
`p = p + phi`. This difference should be considered when comparing the
code against other projection-method formulations.

---

## Common Poisson Solver Interface

The main time-stepping loop always uses the same function call:

```python
phi = solve_poisson(f, dx, dy, config, initial=phi)
```

[`src/lid_driven/poisson/base.py`](src/lid_driven/poisson/base.py)
selects the implementation based on `config.poisson_solver`.

### DCT

File: [`src/lid_driven/poisson/dct.py`](src/lid_driven/poisson/dct.py)

Procedure:

1. Remove the mean of the interior RHS.
2. Apply a type-II discrete cosine transform.
3. Divide by the discrete-Laplacian eigenvalues in transform space.
4. Remove the zero mode.
5. Apply the inverse DCT.
6. Apply homogeneous Neumann boundary conditions.

DCT is a spectral direct method rather than an iterative method. It
requires SciPy.

### SOR

File: [`src/lid_driven/poisson/sor.py`](src/lid_driven/poisson/sor.py)

SOR applies a relaxation factor to the Gauss-Seidel update:

```text
phi_new = (1 - omega) * phi_old + omega * phi_GS
```

Current settings:

```python
sor_omega: float = 1.7
sor_tolerance: float = 1.0e-5
sor_max_iter: int = 1000
```

The SOR stopping criterion is based on the maximum solution update. It
is not the same as a residual norm.

### Multigrid

File:
[`src/lid_driven/poisson/multigrid.py`](src/lid_driven/poisson/multigrid.py)

The geometric V-cycle is:

```text
Pre-smoothing
    -> Compute residual
    -> Restrict to a coarse grid
    -> Solve the coarse-grid error equation
    -> Prolong to the fine grid
    -> Apply correction
    -> Post-smoothing
```

The smoother is weighted Jacobi. Iteration stops when the maximum
residual is below the configured tolerance.

### BiCGSTAB

File:
[`src/lid_driven/poisson/bicgstab.py`](src/lid_driven/poisson/bicgstab.py)

The discrete Laplacian matrix is not assembled explicitly.
`_apply_laplacian()` performs matrix-vector products, making this a
matrix-free Krylov solver.

The Neumann Poisson problem has a constant null space. The
implementation therefore:

- Projects the RHS and solution onto the zero-mean subspace
- Removes the mean of residuals and search directions during iteration
- Uses scale-aware tests for `rho`, `alpha`, and `omega` breakdowns
- Restarts BiCGSTAB from the current residual after a numerical
  breakdown

A `RuntimeError` is raised after repeated breakdowns or when the maximum
iteration count is exceeded.

Current settings:

```python
bicgstab_tolerance: float = 1.0e-5
bicgstab_max_iter: int = 10000
```

The current BiCGSTAB implementation is a pure NumPy, matrix-free
implementation and can be considerably slower than DCT.

---

## Convergence Criterion

The velocity convergence error is:

```text
error_u = ||u^(n+1) - u^n||_F
error_v = ||v^(n+1) - v^n||_F

total_error = sqrt(error_u^2 + error_v^2)
```

The simulation stops normally when:

```python
timestep > 5 and total_error < config.velocity_tolerance
```

The current tolerance is:

```python
velocity_tolerance: float = 1.0e-5
```

The following information is printed every 100 time steps:

```text
[Step     100] max|u| = ..., max|v| = ..., error = ...
```

The `pressure_tolerance` configuration value is not currently used by
the main time-stepping loop. Each Poisson solver uses its own tolerance.

---

## Checkpoints and Safe Interruption

The current checkpoint setting is:

```python
checkpoint_interval: int = 1000
```

The current result is saved to the normal result path every 1,000 time
steps:

```text
Checkpoint saved at step 999: ...
Checkpoint saved at step 1999: ...
```

Set the interval to zero to disable periodic checkpoints:

```python
checkpoint_interval: int = 0
```

### Saving before convergence

Press `Ctrl+C` once in the simulation terminal.

The solver catches `KeyboardInterrupt`, then:

1. Returns the arrays from the latest completed time step.
2. Records `interrupted=True`.
3. Saves the result to the normal output path.

Closing the terminal forcibly or terminating Python through Task
Manager can discard all in-memory progress since the most recent
checkpoint.

---

## Result Files

Output paths are generated by
[`src/lid_driven/output.py`](src/lid_driven/output.py).

The case-name format is:

```text
Re<Reynolds number>_<Poisson solver>
```

Examples:

```text
Re100_dct
Re100_sor
Re500_bicgstab
```

Example data path:

```text
results/data/solution_Re500_bicgstab.npz
```

Saved fields:

| Key | Description |
|---|---|
| `u`, `v` | Staggered velocity fields |
| `p` | Pressure field |
| `phi` | Pressure-correction field |
| `error_history` | Velocity error for each time step |
| `final_timestep` | Index of the final completed time step |
| `x`, `y` | Plotting coordinates |
| `nx`, `ny` | Grid parameters |
| `dx`, `dy` | Grid spacing |
| `re`, `dt` | Simulation parameters |
| `poisson_solver` | Selected Poisson solver |
| `interrupted` | Whether the run was saved after `Ctrl+C` |

Results are first written to a temporary `.tmp.npz` file and then moved
into place with `os.replace()`. This reduces the chance of corrupting an
existing checkpoint if saving is interrupted.

---

## Post-Processing and Figures

[`scripts/plot_results.py`](scripts/plot_results.py) loads the `.npz`
file and calls the post-processing functions.

### Staggered-to-collocated conversion

[`src/lid_driven/postprocess.py`](src/lid_driven/postprocess.py)
converts the staggered fields onto a common plotting grid:

```text
ucol = average of neighboring u values
vcol = average of neighboring v values
pcol = average of neighboring pressure values
```

It also computes:

```text
vorticity = dv/dx - du/dy
velocity magnitude = sqrt(u^2 + v^2)
```

### Generated figures

[`src/lid_driven/plotting.py`](src/lid_driven/plotting.py) saves the
following 300-DPI PNG files:

```text
velocity_magnitude_Re500_bicgstab.png
pressure_Re500_bicgstab.png
vorticity_contour_Re500_bicgstab.png
streamlines_Re500_bicgstab.png
convergence_Re500_bicgstab.png
```

The files are saved in:

```text
results/figures/Re500_bicgstab/
```

Each plotting function calls `plt.show()`. In a GUI environment, close
the current figure window to allow the script to continue to the next
figure.

---

## Comparing Poisson Solvers

Run all solvers at the same Reynolds number:

```powershell
python scripts/run_cfd.py dct
python scripts/run_cfd.py sor
python scripts/run_cfd.py multigrid
python scripts/run_cfd.py bicgstab
```

Generate all figures:

```powershell
python scripts/plot_results.py dct
python scripts/plot_results.py sor
python scripts/plot_results.py multigrid
python scripts/plot_results.py bicgstab
```

The case names differ, so the output files do not overwrite one another.

---

## Project Structure

```text
scripts/
    run_cfd.py              Simulation entry point and NPZ output
    plot_results.py         NPZ post-processing and figure generation

src/lid_driven/
    config.py               Physical and numerical configuration
    grid.py                 Coordinates and staggered-field initialization
    boundary.py             Velocity boundary conditions
    solver.py               Main time-stepping loop
    operators.py            Momentum, TDMA sweeps, projection, pressure
    linalg.py               Thomas algorithm
    diagnostics.py          Divergence, velocity error, maximum velocity
    output.py               Data and figure path generation
    postprocess.py          Collocated conversion, vorticity, speed
    plotting.py             Matplotlib figures
    poisson/
        base.py             Common Poisson dispatcher
        dct.py              DCT solver
        sor.py              SOR solver
        multigrid.py        Geometric multigrid solver
        bicgstab.py         Matrix-free BiCGSTAB solver

tests/
    test_poisson_dispatch.py
    test_output_paths.py
    test_dct.py
    test_sor.py
    test_multigrid.py
    test_tdma.py

results/
    data/                    NPZ results
    figures/                 Case-specific PNG results
```

Generated files inside `results/` are excluded by `.gitignore`.
`.gitkeep` files preserve the `data/` and `figures/` directory structure
in Git.

---

## Tests

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Git Bash:

```bash
PYTHONPATH=src python -m pytest -q
```

The current tests cover:

- Poisson solver dispatch
- Output-path generation from Reynolds number and solver name
- Returned field shape and homogeneous Neumann boundaries
- Error handling for unknown solver names

---

## Numerical Considerations

1. Higher Reynolds numbers may require a smaller `dt` and a finer grid.
2. The previous velocity is initialized to zero at the first time step;
   there is no separate Adams-Bashforth startup scheme.
3. The SOR tolerance is based on the maximum solution update rather
   than the Poisson residual.
4. The Neumann Poisson problem is singular up to an arbitrary constant,
   so the mean is removed from the RHS and solution.
5. `pressure_tolerance` is currently unused; adjust the tolerance of
   the selected Poisson solver instead.
6. The pressure update differs from a simple cumulative projection
   method.
7. Checkpoints overwrite the same case file; previous checkpoint
   generations are not retained.
8. Grid independence, time-step sensitivity, divergence residuals, and
   benchmark comparisons should be evaluated before treating a result
   as validated.

## Reference

Ghia, U., Ghia, K. N., and Shin, C. T. (1982). High-Re solutions for
incompressible flow using the Navier-Stokes equations and a multigrid
method. *Journal of Computational Physics*.
