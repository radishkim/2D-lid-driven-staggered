import numpy as np
from scipy.fft import dctn, idctn


def solve_poisson_dct_neumann(f, dx, dy):
    """
    Solve the pressure Poisson equation with homogeneous Neumann
    boundary conditions using DCT.

    The equation is approximately:

        ∇² phi = f

    Parameters
    ----------
    f : ndarray
        Right-hand side of the Poisson equation.
        Shape: (nx, ny)

    dx, dy : float
        Grid spacing.

    Returns
    -------
    phi : ndarray
        Pressure correction field.
        Shape: (nx, ny)
    """

    nx, ny = f.shape

    phi = np.zeros_like(f)

    # Interior RHS
    rhs = f[1:-1, 1:-1].copy()

    # Compatibility condition for Neumann Poisson problem
    rhs -= rhs.mean()

    # Forward DCT
    rhs_hat = dctn(rhs, type=2, norm="ortho")

    ni, nj = rhs.shape

    kx = np.arange(ni).reshape(-1, 1)
    ky = np.arange(nj).reshape(1, -1)

    eigenvalue = (
        (2.0 * np.cos(np.pi * kx / (ni - 1)) - 2.0) / dx**2
        + (2.0 * np.cos(np.pi * ky / (nj - 1)) - 2.0) / dy**2
    )

    # Remove singularity at zero mode
    eigenvalue[0, 0] = 1.0

    phi_hat = rhs_hat / eigenvalue

    # Pressure is defined up to a constant
    phi_hat[0, 0] = 0.0

    # Inverse DCT
    phi_inner = idctn(phi_hat, type=2, norm="ortho")

    phi[1:-1, 1:-1] = phi_inner

    # Homogeneous Neumann boundary condition
    phi[0, :] = phi[1, :]
    phi[-1, :] = phi[-2, :]
    phi[:, 0] = phi[:, 1]
    phi[:, -1] = phi[:, -2]

    return phi