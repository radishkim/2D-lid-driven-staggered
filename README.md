# 2D Lid-Driven Cavity Flow Solver using a Staggered Grid

This repository contains a Python implementation of a two-dimensional incompressible lid-driven cavity flow solver. The solver is based on a finite difference method on a staggered grid, where the pressure and velocity components are stored at different locations to avoid pressure-velocity decoupling.

The current implementation focuses on the classical lid-driven cavity flow at low Reynolds number. The main objective of this project is to build a clean and extensible CFD solver structure, where the pressure Poisson equation can be solved using different numerical methods such as SOR, DCT-based direct solver, and multigrid methods.

## 1. Overview

The lid-driven cavity problem is a standard benchmark problem for incompressible flow solvers. A square cavity is filled with an incompressible fluid, and the top wall moves with a constant horizontal velocity while the other walls remain stationary.

This project aims to demonstrate the implementation of a CFD solver from scratch, including:

* staggered-grid discretization
* finite difference approximation
* projection method for pressure-velocity coupling
* pressure Poisson equation
* SOR-based iterative solver
* post-processing of velocity, vorticity, and streamfunction fields

## 2. Features

Current features:

* Two-dimensional incompressible Navier-Stokes solver
* Staggered-grid finite difference method
* Projection method
* SOR solver for the pressure Poisson equation
* TDMA-based implicit treatment for viscous terms
* Vorticity and streamfunction calculation
* Basic visualization of flow fields

Planned features:

* DCT-based Poisson solver
* Multigrid Poisson solver
* Comparison of Poisson solver performance
* Validation against benchmark data
* Extension to higher Reynolds numbers
* Possible extension to LES-based cavity flow simulation

## 3. Numerical Method

The governing equations are the two-dimensional incompressible Navier-Stokes equations:

[
\nabla \cdot \mathbf{u} = 0
]

[
\frac{\partial \mathbf{u}}{\partial t}

* \mathbf{u} \cdot \nabla \mathbf{u}
  = -\nabla p
* \frac{1}{Re} \nabla^2 \mathbf{u}
  ]

A projection method is used to enforce the incompressibility constraint. The velocity field is first advanced without the pressure correction, and then a pressure Poisson equation is solved to project the intermediate velocity field onto a divergence-free space.

The solver uses a staggered grid arrangement:

* pressure is stored at cell centers
* horizontal velocity is stored at vertical cell faces
* vertical velocity is stored at horizontal cell faces

This arrangement helps avoid checkerboard pressure modes.

## 4. Poisson Solver Options

The pressure Poisson equation is the main linear system solved in this project. The final goal is to make the Poisson solver interchangeable.

Planned solver options:

| Solver    | Status      | Description                                   |
| --------- | ----------- | --------------------------------------------- |
| SOR       | Implemented | Successive Over-Relaxation iterative method   |
| DCT       | Planned     | Direct solver using discrete cosine transform |
| Multigrid | Planned     | Iterative solver using grid hierarchy         |

The target interface is:

```python
phi = poisson_solver.solve(rhs, phi)
```

This design allows the main CFD solver to remain unchanged when switching between SOR, DCT, and multigrid methods.

## 5. Project Structure

```text
lid-driven-staggered/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── lid_driven/
│       ├── __init__.py
│       ├── config.py
│       ├── grid.py
│       ├── boundary.py
│       ├── linalg.py
│       ├── diagnostics.py
│       ├── solver.py
│       └── poisson/
│           ├── __init__.py
│           ├── base.py
│           ├── sor.py
│           ├── dct.py
│           └── multigrid.py
├── scripts/
│   ├── run_cavity.py
│   └── plot_results.py
├── notebooks/
├── results/
│   ├── figures/
│   └── data/
├── docs/
│   ├── numerical_method.md
│   ├── poisson_solvers.md
│   └── validation.md
└── tests/
    ├── test_tdma.py
    ├── test_sor.py
    ├── test_dct.py
    └── test_multigrid.py
```

## 6. How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the simulation:

```bash
python scripts/run_cavity.py
```

Run the simulation with a specific Poisson solver:

```bash
python scripts/run_cavity.py --poisson sor
```

Planned options:

```bash
python scripts/run_cavity.py --poisson dct
python scripts/run_cavity.py --poisson multigrid
```

## 7. Results

Example results will be added in this section, including:

* velocity vector field
* streamfunction contour
* vorticity contour
* centerline velocity profiles

## 8. Validation

The solver will be validated against benchmark data for the two-dimensional lid-driven cavity flow.

Planned validation:

* comparison of horizontal velocity along the vertical centerline
* comparison of vertical velocity along the horizontal centerline
* benchmark comparison with Ghia et al. (1982)

## 9. Future Work

* Refactor the current SOR-based solver into a modular Poisson solver interface
* Add DCT-based pressure Poisson solver
* Add multigrid pressure Poisson solver
* Compare convergence speed and computational cost of SOR, DCT, and multigrid methods
* Add automated tests for TDMA and Poisson solvers
* Extend the solver to higher Reynolds numbers
* Investigate LES modeling for cavity flow

## 10. References

* Ghia, U., Ghia, K. N., and Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. Journal of Computational Physics.
