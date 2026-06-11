# 2D Lid-Driven Cavity Flow Solver

A Python CFD solver for two-dimensional incompressible lid-driven cavity
flow. The code uses a staggered finite-difference grid, a projection
method for pressure-velocity coupling, and interchangeable pressure
Poisson solvers.

## Features

- Two-dimensional incompressible Navier-Stokes equations
- Staggered-grid finite-difference discretization
- Projection method
- TDMA-based implicit viscous treatment
- Interchangeable DCT, SOR, multigrid, and BiCGSTAB Poisson solvers
- Automatic `.npz` result naming from Reynolds number and solver name
- Automatic figure input selection, directory creation, and naming
- Velocity magnitude, pressure, vorticity, streamline, and convergence plots

## Project Structure

```text
src/lid_driven/
    config.py             Simulation parameters
    solver.py             Main time-marching solver
    output.py             Shared result and figure path rules
    plotting.py           Figure generation functions
    postprocess.py        Staggered-to-collocated conversion
    poisson/
        base.py           Common Poisson solver dispatcher
        dct.py            DCT solver
        sor.py            SOR solver
        multigrid.py      Geometric multigrid solver
        bicgstab.py       BiCGSTAB solver

scripts/
    run_cfd.py            Simulation entry point
    plot_results.py       Post-processing and plotting entry point

results/
    data/                 Saved `.npz` simulation results
    figures/              Case-specific figure directories

tests/                    Unit tests
docs/                     Numerical-method documentation
```

`run_cfd.py` reads the current Reynolds number from `SimulationConfig`
and is not restricted to a specific Reynolds number.

## Installation

From the project root, create and activate a virtual environment if
needed, then install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The DCT solver requires SciPy. It is included in `requirements.txt`.

## Basic Configuration

### 1. Select the Reynolds number

Edit `src/lid_driven/config.py`:

```python
@dataclass
class SimulationConfig:
    re: float = 1000.0
```

Any positive integer Reynolds number can be used, for example:

```python
re: float = 10.0
re: float = 100.0
re: float = 400.0
re: float = 1000.0
```

Grid size, time step, maximum time steps, and convergence tolerances are
configured in the same class:

```python
nx: int = 130
ny: int = 130
dt: float = 0.005
max_timestep: int = 1_000_000
checkpoint_interval: int = 1000
velocity_tolerance: float = 1.0e-5
```

Higher Reynolds numbers can require a smaller time step, a finer grid,
or more time steps. Changing only `re` changes the physical parameter,
but does not guarantee that the existing numerical settings are
appropriate for every Reynolds number.

`checkpoint_interval` controls how often the current solution is saved.
With the default value, the `.npz` result file is updated every 1000
time steps. Set it to `0` to disable periodic checkpoint saves.

### 2. Select the Poisson solver

Edit `scripts/run_cfd.py`:

```python
POISSON_SOLVER = "dct"
```

Available values are:

```python
POISSON_SOLVER = "dct"
POISSON_SOLVER = "sor"
POISSON_SOLVER = "multigrid"
POISSON_SOLVER = "bicgstab"
```

Solver-specific tolerances and iteration limits are stored in
`src/lid_driven/config.py`.

## Running A Simulation

The standard workflow requires only two configuration changes:

1. Set `re` in `src/lid_driven/config.py`.
2. Set `POISSON_SOLVER` in `scripts/run_cfd.py`.
3. Run the simulation.
4. Run the plotting script.

For example, to calculate `Re=1000` using SOR:

1. In `src/lid_driven/config.py`, set:

```python
re: float = 1000.0
```

2. In `scripts/run_cfd.py`, set:

```python
POISSON_SOLVER = "sor"
```

3. Run:

```powershell
python scripts/run_cfd.py
```

The data file is selected automatically:

```text
results/data/solution_Re1000_sor.npz
```

The `.npz` file contains:

- `u`, `v`: staggered velocity fields
- `p`: pressure field
- `phi`: pressure correction
- `error_history`: velocity convergence history
- `final_timestep`: final time-step index
- `x`, `y`: plotting coordinates
- `nx`, `ny`, `dx`, `dy`: grid information
- `re`, `dt`: simulation parameters
- `poisson_solver`: selected Poisson solver

## Generating Figures

After the simulation finishes, run:

```powershell
python scripts/plot_results.py
```

`plot_results.py` automatically reads:

- `re` from `src/lid_driven/config.py`
- `POISSON_SOLVER` from `scripts/run_cfd.py`

For the `Re=1000`, SOR example, it loads:

```text
results/data/solution_Re1000_sor.npz
```

and creates:

```text
results/figures/Re1000_sor/
    velocity_magnitude_Re1000_sor.png
    pressure_Re1000_sor.png
    vorticity_contour_Re1000_sor.png
    streamlines_Re1000_sor.png
    convergence_Re1000_sor.png
```

Results from different Reynolds numbers or Poisson solvers are stored
separately and do not overwrite one another.

## Saving Before Convergence

The simulation does not need to reach `velocity_tolerance` before its
current state can be saved.

- The result `.npz` file is updated every `checkpoint_interval` steps.
- Press `Ctrl+C` once in the simulation terminal to stop safely.
- The solver catches the interrupt, returns the latest completed step,
  and writes it to the normal result path.
- Do not close the terminal or terminate the Python process from Task
  Manager if the latest in-memory step must be preserved.

For example, an interrupted `Re=500` SOR calculation is saved as:

```text
results/data/solution_Re500_sor.npz
```

The file contains `interrupted=True` when it was saved after `Ctrl+C`.
It can be plotted normally:

```powershell
python scripts/plot_results.py sor
```

## Command-Line Solver Override

The value of `POISSON_SOLVER` is the default. A solver can also be
selected temporarily from the command line without editing the script:

```powershell
python scripts/run_cfd.py dct
python scripts/run_cfd.py sor
python scripts/run_cfd.py multigrid
python scripts/run_cfd.py bicgstab
```

Use the same solver argument when plotting:

```powershell
python scripts/plot_results.py dct
python scripts/plot_results.py sor
python scripts/plot_results.py multigrid
python scripts/plot_results.py bicgstab
```

The Reynolds number still comes from `SimulationConfig`.

## Comparing Multiple Poisson Solvers

To compare solvers at the same Reynolds number:

```powershell
python scripts/run_cfd.py dct
python scripts/run_cfd.py sor
python scripts/run_cfd.py multigrid
python scripts/run_cfd.py bicgstab

python scripts/plot_results.py dct
python scripts/plot_results.py sor
python scripts/plot_results.py multigrid
python scripts/plot_results.py bicgstab
```

For `Re=100`, the data files are:

```text
results/data/solution_Re100_dct.npz
results/data/solution_Re100_sor.npz
results/data/solution_Re100_multigrid.npz
results/data/solution_Re100_bicgstab.npz
```

## Poisson Solver Interface

The main CFD loop does not contain solver-specific branches. It always
uses the common dispatcher:

```python
phi = solve_poisson(f, dx, dy, config, initial=phi)
```

`src/lid_driven/poisson/base.py` selects the implementation using
`config.poisson_solver`.

## Tests

Run the test suite from the project root:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

## References

- Ghia, U., Ghia, K. N., and Shin, C. T. (1982). High-Re solutions for
  incompressible flow using the Navier-Stokes equations and a multigrid
  method. Journal of Computational Physics.
