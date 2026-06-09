import numpy as np

import numpy as np


def staggered_to_collocated(u, v, p, nx, ny):
    """
    Convert staggered-grid variables to collocated-grid variables
    for visualization and validation.

    Parameters
    ----------
    u : ndarray
        Staggered x-velocity.
        Shape: (nx - 1, ny)

    v : ndarray
        Staggered y-velocity.
        Shape: (nx, ny - 1)

    p : ndarray
        Pressure field.
        Shape: (nx, ny)

    nx, ny : int
        Grid parameters.

    Returns
    -------
    ucol, vcol, pcol : ndarray
        Collocated variables.
        Shape: (nx - 1, ny - 1)
    """

    ucol = np.zeros((nx - 1, ny - 1))
    vcol = np.zeros((nx - 1, ny - 1))
    pcol = np.zeros((nx - 1, ny - 1))

    # u is located on vertical faces.
    # Average two neighboring u values in y-direction.
    for j in range(ny - 1):
        ucol[:, j] = 0.5 * (u[:, j] + u[:, j + 1])

    # v is located on horizontal faces.
    # Average two neighboring v values in x-direction.
    for i in range(nx - 1):
        vcol[i, :] = 0.5 * (v[i, :] + v[i + 1, :])

    # Pressure is stored on cell centers with ghost cells.
    # Convert to plotting grid by averaging neighboring pressure values.
    pcol[0, :] = 0.0
    pcol[:, 0] = 0.0
    pcol[nx - 2, :] = 0.0
    pcol[:, ny - 2] = 0.0

    for i in range(1, nx - 2):
        for j in range(1, ny - 2):
            pcol[i, j] = 0.25 * (
                p[i, j]
                + p[i + 1, j]
                + p[i, j + 1]
                + p[i + 1, j + 1]
            )

    return ucol, vcol, pcol


def compute_vorticity(ucol, vcol, dx, dy):
    """
    Compute vorticity from collocated velocity fields.

        omega = dv/dx - du/dy

    Parameters
    ----------
    ucol, vcol : ndarray
        Collocated velocity fields.

    dx, dy : float
        Grid spacing.

    Returns
    -------
    omega : ndarray
        Vorticity field.
    """

    nx, ny = ucol.shape
    omega = np.zeros_like(ucol)

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            dvdx = (vcol[i + 1, j] - vcol[i - 1, j]) / (2.0 * dx)
            dudy = (ucol[i, j + 1] - ucol[i, j - 1]) / (2.0 * dy)

            omega[i, j] = dvdx - dudy

    return omega


def compute_velocity_magnitude(ucol, vcol):
    """
    Compute velocity magnitude.

        |U| = sqrt(u^2 + v^2)
    """

    return np.sqrt(ucol**2 + vcol**2)