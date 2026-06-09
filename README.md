# 2D Lid-Driven Cavity Flow Solver using a Staggered Grid

A Python CFD solver for the two-dimensional lid-driven cavity flow using a staggered grid, projection method, and interchangeable Poisson solvers.

## Overview

This project implements a two-dimensional incompressible lid-driven cavity flow solver. The solver is based on a finite difference method on a staggered grid and uses a projection method for pressure-velocity coupling.

The first implementation focuses on the SOR method for solving the pressure Poisson equation. The code structure is designed to be extended later with DCT-based and multigrid Poisson solvers.

## Main Features

Current features:

- Two-dimensional incompressible Navier-Stokes solver
- Staggered-grid finite difference method
- Projection method
- SOR pressure Poisson solver
- TDMA-based implicit treatment for viscous terms
- Vorticity and streamfunction post-processing

Planned features:

- DCT-based Poisson solver
- Multigrid Poisson solver
- Comparison of Poisson solver performance
- Validation against benchmark data
- Extension to higher Reynolds numbers

## Project Structure

```text
src/lid_driven/          Main solver package
src/lid_driven/poisson/  Pressure Poisson solvers
scripts/                 Execution and plotting scripts
notebooks/               Original notebooks and experiments
results/                 Simulation results
docs/                    Numerical method notes
tests/                   Unit tests
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the simulation:

```bash
python scripts/run_cavity.py
```

After the solver is modularized, the Poisson solver will be selected from the command line:

```bash
python scripts/run_cavity.py --poisson sor
python scripts/run_cavity.py --poisson dct
python scripts/run_cavity.py --poisson multigrid
```

## Poisson Solver Plan

The final goal is to make the pressure Poisson solver interchangeable without changing the main CFD solver.

Planned options:

| Solver | Status | Description |
|---|---|---|
| SOR | In progress | Successive Over-Relaxation method |
| DCT | Planned | Direct Poisson solver using discrete cosine transform |
| Multigrid | Planned | Iterative solver using grid hierarchy |

The target interface is:

```python
phi = poisson_solver.solve(rhs, phi_initial)
```

With this structure, the main solver can call the same `solve()` method regardless of whether the pressure Poisson equation is solved using SOR, DCT, or multigrid.

## Development Roadmap

1. Refactor the existing TDMA routine into `src/lid_driven/linalg.py`
2. Move the SOR pressure Poisson solver into `src/lid_driven/poisson/sor.py`
3. Define a common Poisson solver interface in `src/lid_driven/poisson/base.py`
4. Connect the Poisson solver option to `scripts/run_cavity.py`
5. Add DCT and multigrid solvers
6. Validate the solver against benchmark lid-driven cavity data

## References

- Ghia, U., Ghia, K. N., and Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. Journal of Computational Physics.