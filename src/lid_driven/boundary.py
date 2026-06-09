def apply_velocity_bc(u, v, u_lid, nx, ny):
    """
    Apply velocity boundary conditions for the 2D lid-driven cavity
    on a staggered grid.

    Parameters
    ----------
    u : ndarray
        x-direction velocity on staggered grid.
        Shape: (nx - 1, ny)

    v : ndarray
        y-direction velocity on staggered grid.
        Shape: (nx, ny - 1)

    u_lid : float
        Lid velocity at the top wall.

    nx, ny : int
        Grid parameters.

    Returns
    -------
    u, v : ndarray
        Velocity fields after applying boundary conditions.
    """

    # ------------------------------------------------------------
    # u-velocity boundary conditions
    # ------------------------------------------------------------

    # Left wall: u = 0
    u[0, :] = 0.0

    # Right wall: u = 0
    u[nx - 2, :] = 0.0

    # Bottom wall: no-slip condition using ghost value
    u[:, 0] = -u[:, 1]

    # Top wall: moving lid condition using ghost value
    u[:, ny - 1] = 2.0 * u_lid - u[:, ny - 2]

    # ------------------------------------------------------------
    # v-velocity boundary conditions
    # ------------------------------------------------------------

    # Bottom wall: v = 0
    v[:, 0] = 0.0

    # Top wall: v = 0
    v[:, ny - 2] = 0.0

    # Left wall: no-slip condition using ghost value
    v[0, :] = -v[1, :]

    # Right wall: no-slip condition using ghost value
    v[nx - 1, :] = -v[nx - 2, :]

    return u, v