import numpy as np


def _apply_laplacian(phi, dx, dy):
    padded = np.pad(phi, 1, mode="edge")
    return (
        (padded[2:, 1:-1] - 2.0 * phi + padded[:-2, 1:-1]) / dx**2
        + (padded[1:-1, 2:] - 2.0 * phi + padded[1:-1, :-2]) / dy**2
    )


def _smooth(phi, rhs, dx, dy, omega, iterations):
    denominator = 2.0 / dx**2 + 2.0 / dy**2

    for _ in range(iterations):
        padded = np.pad(phi, 1, mode="edge")
        jacobi = (
            (padded[2:, 1:-1] + padded[:-2, 1:-1]) / dx**2
            + (padded[1:-1, 2:] + padded[1:-1, :-2]) / dy**2
            - rhs
        ) / denominator
        phi = (1.0 - omega) * phi + omega * jacobi
        phi -= phi.mean()

    return phi


def _restrict(fine):
    rows = (fine.shape[0] + 1) // 2
    cols = (fine.shape[1] + 1) // 2
    coarse = np.empty((rows, cols), dtype=fine.dtype)

    for i in range(rows):
        for j in range(cols):
            block = fine[2 * i : min(2 * i + 2, fine.shape[0]),
                         2 * j : min(2 * j + 2, fine.shape[1])]
            coarse[i, j] = block.mean()

    coarse -= coarse.mean()
    return coarse


def _prolong(coarse, shape):
    coarse_x = np.linspace(0.0, 1.0, coarse.shape[0])
    coarse_y = np.linspace(0.0, 1.0, coarse.shape[1])
    fine_x = np.linspace(0.0, 1.0, shape[0])
    fine_y = np.linspace(0.0, 1.0, shape[1])

    in_x = np.empty((shape[0], coarse.shape[1]), dtype=coarse.dtype)
    for j in range(coarse.shape[1]):
        in_x[:, j] = np.interp(fine_x, coarse_x, coarse[:, j])

    fine = np.empty(shape, dtype=coarse.dtype)
    for i in range(shape[0]):
        fine[i, :] = np.interp(fine_y, coarse_y, in_x[i, :])

    fine -= fine.mean()
    return fine


def _v_cycle(
    phi,
    rhs,
    dx,
    dy,
    pre_smooth,
    post_smooth,
    omega,
):
    if min(rhs.shape) <= 4:
        return _smooth(phi, rhs, dx, dy, omega, 50)

    phi = _smooth(phi, rhs, dx, dy, omega, pre_smooth)
    residual = rhs - _apply_laplacian(phi, dx, dy)
    coarse_rhs = _restrict(residual)

    coarse_error = np.zeros_like(coarse_rhs)
    coarse_error = _v_cycle(
        coarse_error,
        coarse_rhs,
        2.0 * dx,
        2.0 * dy,
        pre_smooth,
        post_smooth,
        omega,
    )

    phi += _prolong(coarse_error, phi.shape)
    phi -= phi.mean()
    return _smooth(phi, rhs, dx, dy, omega, post_smooth)


def solve_poisson_multigrid(
    rhs,
    dx,
    dy,
    tolerance,
    max_cycles,
    pre_smooth=3,
    post_smooth=3,
    omega=0.8,
    initial=None,
):
    """Solve a homogeneous-Neumann Poisson problem with geometric multigrid."""
    inner_rhs = rhs[1:-1, 1:-1].copy()
    inner_rhs -= inner_rhs.mean()

    if initial is None:
        phi_inner = np.zeros_like(inner_rhs)
    else:
        phi_inner = initial[1:-1, 1:-1].copy()
        phi_inner -= phi_inner.mean()

    for _ in range(max_cycles):
        phi_inner = _v_cycle(
            phi_inner,
            inner_rhs,
            dx,
            dy,
            pre_smooth,
            post_smooth,
            omega,
        )
        residual = inner_rhs - _apply_laplacian(phi_inner, dx, dy)
        if np.max(np.abs(residual)) < tolerance:
            break

    phi = np.zeros_like(rhs)
    phi[1:-1, 1:-1] = phi_inner
    phi[0, :] = phi[1, :]
    phi[-1, :] = phi[-2, :]
    phi[:, 0] = phi[:, 1]
    phi[:, -1] = phi[:, -2]
    return phi
