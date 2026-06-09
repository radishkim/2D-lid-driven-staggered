import numpy as np


def create_grid(config):
    """
    Create x and y coordinates for post-processing.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration object.

    Returns
    -------
    x : ndarray
        x-coordinate array.
    y : ndarray
        y-coordinate array.
    """
    x = np.linspace(config.x0, config.xmax, config.nx - 1)
    y = np.linspace(config.y0, config.ymax, config.ny - 1)

    return x, y


def initialize_fields(config):
    """
    Initialize staggered-grid velocity and pressure fields.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration object.

    Returns
    -------
    fields : dict
        Dictionary containing velocity, pressure, and auxiliary arrays.
    """
    nx = config.nx
    ny = config.ny

    # Pressure-related variables
    phi = np.zeros((nx, ny))
    p = np.zeros_like(phi)

    # Staggered velocity variables
    u = np.zeros((nx - 1, ny))
    v = np.zeros((nx, ny - 1))

    # Previous time-step velocities
    up = np.zeros_like(u)
    vp = np.zeros_like(v)

    # Velocity correction terms from implicit diffusion solve
    du = np.zeros_like(u)
    dv = np.zeros_like(v)

    # Corrected velocities after projection
    un = np.zeros_like(u)
    vn = np.zeros_like(v)

    # Intermediate velocities
    uhat = np.zeros_like(u)
    vhat = np.zeros_like(v)

    fields = {
        "phi": phi,
        "p": p,
        "u": u,
        "v": v,
        "up": up,
        "vp": vp,
        "du": du,
        "dv": dv,
        "un": un,
        "vn": vn,
        "uhat": uhat,
        "vhat": vhat,
    }

    return fields