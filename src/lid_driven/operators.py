import numpy as np
from .linalg import tdma


def compute_u_rhs(u, v, up, vp, dx, dy, dt, re):
    """
    Compute the right-hand side of the u-momentum equation.

    The convection term is treated by Adams-Bashforth 2nd order,
    and the diffusion term is included explicitly as part of the
    ADI/Crank-Nicolson splitting procedure.

    Parameters
    ----------
    u, v : ndarray
        Current velocity fields.

    up, vp : ndarray
        Previous time-step velocity fields.

    dx, dy : float
        Grid spacing.

    dt : float
        Time-step size.

    re : float
        Reynolds number.

    Returns
    -------
    rhsu : ndarray
        Right-hand side for the u-momentum equation.
        Same shape as u.
    """
    nx_minus_1, ny = u.shape
    nx = nx_minus_1 + 1

    rhsu = np.zeros_like(u)

    for ix in range(1, nx - 2):
        for iy in range(1, ny - 1):
            # --------------------------------------------------------
            # Current-step nonlinear convection terms
            # --------------------------------------------------------

            # d(u^2)/dx term
            ur = 0.5 * (u[ix + 1, iy] ** 2 + u[ix, iy] ** 2)
            ul = 0.5 * (u[ix, iy] ** 2 + u[ix - 1, iy] ** 2)

            # d(uv)/dy term
            uvu = 0.25 * (u[ix, iy + 1] + u[ix, iy]) * (
                v[ix + 1, iy] + v[ix, iy]
            )

            uvd = 0.25 * (u[ix, iy] + u[ix, iy - 1]) * (
                v[ix + 1, iy - 1] + v[ix, iy - 1]
            )

            # --------------------------------------------------------
            # Previous-step nonlinear convection terms
            # --------------------------------------------------------

            urp = 0.5 * (up[ix + 1, iy] ** 2 + up[ix, iy] ** 2)
            ulp = 0.5 * (up[ix, iy] ** 2 + up[ix - 1, iy] ** 2)

            uvup = 0.25 * (up[ix, iy + 1] + up[ix, iy]) * (
                vp[ix + 1, iy] + vp[ix, iy]
            )

            uvdp = 0.25 * (up[ix, iy] + up[ix, iy - 1]) * (
                vp[ix + 1, iy - 1] + vp[ix, iy - 1]
            )

            # --------------------------------------------------------
            # Adams-Bashforth 2nd order convection
            # --------------------------------------------------------

            convection = (
                1.5 * (ur - ul) / dx
                + 1.5 * (uvu - uvd) / dy
                - 0.5 * (urp - ulp) / dx
                - 0.5 * (uvup - uvdp) / dy
            )

            # --------------------------------------------------------
            # Diffusion term
            # --------------------------------------------------------

            diffusion = (
                (u[ix + 1, iy] - 2.0 * u[ix, iy] + u[ix - 1, iy]) / dx**2
                + (u[ix, iy + 1] - 2.0 * u[ix, iy] + u[ix, iy - 1]) / dy**2
            )

            rhsu[ix, iy] = -dt * convection + dt / re * diffusion

    return rhsu




def compute_v_rhs(u, v, up, vp, dx, dy, dt, re):
    """
    Compute the right-hand side of the v-momentum equation.

    The convection term is treated by Adams-Bashforth 2nd order,
    and the diffusion term is included explicitly as part of the
    ADI/Crank-Nicolson splitting procedure.

    Parameters
    ----------
    u, v : ndarray
        Current velocity fields.

    up, vp : ndarray
        Previous time-step velocity fields.

    dx, dy : float
        Grid spacing.

    dt : float
        Time-step size.

    re : float
        Reynolds number.

    Returns
    -------
    rhsv : ndarray
        Right-hand side for the v-momentum equation.
        Same shape as v.
    """
    nx, ny_minus_1 = v.shape
    ny = ny_minus_1 + 1

    rhsv = np.zeros_like(v)

    for ix in range(1, nx - 1):
        for iy in range(1, ny - 2):
            # --------------------------------------------------------
            # Current-step nonlinear convection terms
            # --------------------------------------------------------

            # d(uv)/dx term
            vru = 0.25 * (v[ix + 1, iy] + v[ix, iy]) * (
                u[ix, iy + 1] + u[ix, iy]
            )

            vlu = 0.25 * (v[ix, iy] + v[ix - 1, iy]) * (
                u[ix - 1, iy + 1] + u[ix - 1, iy]
            )

            # d(v^2)/dy term
            vu = 0.5 * (v[ix, iy + 1] ** 2 + v[ix, iy] ** 2)
            vd = 0.5 * (v[ix, iy] ** 2 + v[ix, iy - 1] ** 2)

            # --------------------------------------------------------
            # Previous-step nonlinear convection terms
            # --------------------------------------------------------

            vrup = 0.25 * (vp[ix + 1, iy] + vp[ix, iy]) * (
                up[ix, iy + 1] + up[ix, iy]
            )

            vlup = 0.25 * (vp[ix, iy] + vp[ix - 1, iy]) * (
                up[ix - 1, iy + 1] + up[ix - 1, iy]
            )

            vup = 0.5 * (vp[ix, iy + 1] ** 2 + vp[ix, iy] ** 2)
            vdp = 0.5 * (vp[ix, iy] ** 2 + vp[ix, iy - 1] ** 2)

            # --------------------------------------------------------
            # Adams-Bashforth 2nd order convection
            # --------------------------------------------------------

            convection = (
                1.5 * (vru - vlu) / dx
                + 1.5 * (vu - vd) / dy
                - 0.5 * (vrup - vlup) / dx
                - 0.5 * (vup - vdp) / dy
            )

            # --------------------------------------------------------
            # Diffusion term
            # --------------------------------------------------------

            diffusion = (
                (v[ix + 1, iy] - 2.0 * v[ix, iy] + v[ix - 1, iy]) / dx**2
                + (v[ix, iy + 1] - 2.0 * v[ix, iy] + v[ix, iy - 1]) / dy**2
            )

            rhsv[ix, iy] = -dt * convection + dt / re * diffusion

    return rhsv



def solve_u_diffusion_tdma(rhsu, dx, dy, dt, re):
    """
    Solve the implicit diffusion part for the u-momentum equation
    using ADI-type TDMA sweeps.

    Parameters
    ----------
    rhsu : ndarray
        Right-hand side of the u-momentum equation.
        Shape: (nx - 1, ny)

    dx, dy : float
        Grid spacing.

    dt : float
        Time-step size.

    re : float
        Reynolds number.

    Returns
    -------
    du : ndarray
        Velocity correction for u.
        Same shape as rhsu.
    """
    nx_minus_1, ny = rhsu.shape
    nx = nx_minus_1 + 1

    du = np.zeros_like(rhsu)
    y_sweep_result = np.zeros_like(rhsu)

    betax = dt / (2.0 * re * dx**2)
    betay = dt / (2.0 * re * dy**2)

    # ------------------------------------------------------------
    # First sweep: x-direction TDMA
    # ------------------------------------------------------------
    a = np.ones(nx - 1) * (-betax)
    b = np.ones(nx - 1) * (1.0 + 2.0 * betax)
    c = np.ones(nx - 1) * (-betax)

    for iy in range(1, ny - 1):
        y_sweep_result[:, iy] = tdma(a, b, c, rhsu[:, iy])

    # ------------------------------------------------------------
    # Second sweep: y-direction TDMA
    # ------------------------------------------------------------
    a = np.ones(ny) * (-betay)
    b = np.ones(ny) * (1.0 + 2.0 * betay)
    c = np.ones(ny) * (-betay)

    # Boundary modification due to ghost-cell treatment
    b[1] += betay
    b[-2] += betay

    for ix in range(1, nx - 2):
        du[ix, :] = tdma(a, b, c, y_sweep_result[ix, :])

    return du




def solve_v_diffusion_tdma(rhsv, dx, dy, dt, re):
    """
    Solve the implicit diffusion part for the v-momentum equation
    using ADI-type TDMA sweeps.

    Parameters
    ----------
    rhsv : ndarray
        Right-hand side of the v-momentum equation.
        Shape: (nx, ny - 1)

    dx, dy : float
        Grid spacing.

    dt : float
        Time-step size.

    re : float
        Reynolds number.

    Returns
    -------
    dv : ndarray
        Velocity correction for v.
        Same shape as rhsv.
    """
    nx, ny_minus_1 = rhsv.shape
    ny = ny_minus_1 + 1

    dv = np.zeros_like(rhsv)
    x_sweep_result = np.zeros_like(rhsv)

    betax = dt / (2.0 * re * dx**2)
    betay = dt / (2.0 * re * dy**2)

    # ------------------------------------------------------------
    # First sweep: y-direction TDMA
    # ------------------------------------------------------------
    a = np.ones(ny - 1) * (-betay)
    b = np.ones(ny - 1) * (1.0 + 2.0 * betay)
    c = np.ones(ny - 1) * (-betay)

    for ix in range(1, nx - 1):
        x_sweep_result[ix, :] = tdma(a, b, c, rhsv[ix, :])

    # ------------------------------------------------------------
    # Second sweep: x-direction TDMA
    # ------------------------------------------------------------
    a = np.ones(nx) * (-betax)
    b = np.ones(nx) * (1.0 + 2.0 * betax)
    c = np.ones(nx) * (-betax)

    # Boundary modification due to ghost-cell treatment
    b[1] += betax
    b[-2] += betax

    for iy in range(1, ny - 2):
        dv[:, iy] = tdma(a, b, c, x_sweep_result[:, iy])

    return dv



def project_velocity(uhat, vhat, phi, dx, dy, dt):
    """
    Project intermediate velocity onto a divergence-free velocity field.

    Parameters
    ----------
    uhat, vhat : ndarray
        Intermediate velocity fields.

    phi : ndarray
        Pressure correction field.

    dx, dy : float
        Grid spacing.

    dt : float
        Time step size.

    Returns
    -------
    un, vn : ndarray
        Corrected velocity fields.
    """
    nx = phi.shape[0]
    ny = phi.shape[1]

    un = np.zeros_like(uhat)
    vn = np.zeros_like(vhat)

    # Correct u velocity
    for ix in range(1, nx - 2):
        for iy in range(1, ny - 1):
            un[ix, iy] = uhat[ix, iy] - dt * (
                phi[ix + 1, iy] - phi[ix, iy]
            ) / dx

    # Correct v velocity
    for ix in range(1, nx - 1):
        for iy in range(1, ny - 2):
            vn[ix, iy] = vhat[ix, iy] - dt * (
                phi[ix, iy + 1] - phi[ix, iy]
            ) / dy

    return un, vn




def update_pressure(phi, dx, dy, dt, re):
    """
    Update physical pressure from pressure correction field.

    Parameters
    ----------
    phi : ndarray
        Pressure correction field.

    dx, dy : float
        Grid spacing.

    dt : float
        Time step size.

    re : float
        Reynolds number.

    Returns
    -------
    p : ndarray
        Updated pressure field.
    """
    nx = phi.shape[0]
    ny = phi.shape[1]

    p = np.zeros_like(phi)

    # y-direction contribution
    for iy in range(1, ny - 1):
        p[:, iy] = phi[:, iy] - dt * 0.5 / re * (
            phi[:, iy - 1]
            - 2.0 * phi[:, iy]
            + phi[:, iy + 1]
        ) / dy**2

    # x-direction contribution
    for ix in range(1, nx - 1):
        p[ix, :] += -dt * 0.5 / re * (
            phi[ix - 1, :]
            - 2.0 * phi[ix, :]
            + phi[ix + 1, :]
        ) / dx**2

    return p