import numpy as np


def compute_velocity_error(u, v, up, vp):
    """
    Compute the global velocity difference between two consecutive steps.

    Parameters
    ----------
    u, v : ndarray
        Current velocity fields.

    up, vp : ndarray
        Previous velocity fields.

    Returns
    -------
    total_error : float
        Frobenius-norm based velocity error.
    """
    error_u = np.linalg.norm(u - up, ord="fro")
    error_v = np.linalg.norm(v - vp, ord="fro")

    total_error = np.sqrt(error_u**2 + error_v**2)

    return total_error


def compute_divergence(uhat, vhat, dx, dy, dt):
    """
    Compute the divergence-based RHS for the pressure Poisson equation.

    This function computes:

        f = div(u_hat) / dt

    Parameters
    ----------
    uhat : ndarray
        Intermediate x-direction velocity.
        Shape: (nx - 1, ny)

    vhat : ndarray
        Intermediate y-direction velocity.
        Shape: (nx, ny - 1)

    dx, dy : float
        Grid spacing.

    dt : float
        Time step size.

    Returns
    -------
    f : ndarray
        RHS of the pressure Poisson equation.
        Shape: (nx, ny)
    """
    nx = vhat.shape[0]
    ny = uhat.shape[1]

    f = np.zeros((nx, ny))

    for ix in range(1, nx - 1):
        for iy in range(1, ny - 1):
            f[ix, iy] = (
                (uhat[ix, iy] - uhat[ix - 1, iy]) / dx
                + (vhat[ix, iy] - vhat[ix, iy - 1]) / dy
            ) / dt

    return f


def get_max_velocity(u, v):
    """
    Return maximum absolute velocity components.
    """
    max_u = np.max(np.abs(u))
    max_v = np.max(np.abs(v))

    return max_u, max_v