import numpy as np
import pytest

from lid_driven.config import SimulationConfig
from lid_driven.poisson import solve_poisson


@pytest.mark.parametrize(
    "solver_name",
    ["dct", "sor", "multigrid", "bicgstab"],
)
def test_poisson_dispatch_returns_finite_neumann_solution(solver_name):
    rhs = np.zeros((10, 10))
    rhs[3, 4] = 1.0
    rhs[6, 5] = -1.0
    config = SimulationConfig(
        poisson_solver=solver_name,
        sor_tolerance=1.0e-7,
        multigrid_tolerance=1.0e-7,
        bicgstab_tolerance=1.0e-7,
    )

    phi = solve_poisson(rhs, 0.125, 0.125, config)

    assert phi.shape == rhs.shape
    assert np.all(np.isfinite(phi))
    assert np.allclose(phi[0, :], phi[1, :])
    assert np.allclose(phi[-1, :], phi[-2, :])
    assert np.allclose(phi[:, 0], phi[:, 1])
    assert np.allclose(phi[:, -1], phi[:, -2])


def test_poisson_dispatch_rejects_unknown_solver():
    config = SimulationConfig(poisson_solver="unknown")

    with pytest.raises(ValueError, match="Available solvers"):
        solve_poisson(np.zeros((6, 6)), 0.25, 0.25, config)
