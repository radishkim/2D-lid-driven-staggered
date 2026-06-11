import numpy as np

from .boundary import apply_velocity_bc
from .diagnostics import (
    compute_divergence,
    compute_velocity_error,
    get_max_velocity,
)
from .grid import initialize_fields
from .operators import (
    compute_u_rhs,
    compute_v_rhs,
    solve_u_diffusion_tdma,
    solve_v_diffusion_tdma,
    project_velocity,
    update_pressure,
)
from .poisson import solve_poisson


def run_simulation(config, checkpoint_callback=None):
    """
    Run the 2D lid-driven cavity simulation on a staggered grid.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration object.

    Returns
    -------
    results : dict
        Dictionary containing velocity, pressure, pressure correction,
        error history, and final timestep.
    """

    # ------------------------------------------------------------
    # 1. Load parameters
    # ------------------------------------------------------------
    nx = config.nx
    ny = config.ny
    dx = config.dx
    dy = config.dy
    dt = config.dt
    re = config.re
    u_lid = config.u_lid

    # ------------------------------------------------------------
    # 2. Initialize fields
    # ------------------------------------------------------------
    fields = initialize_fields(config)

    phi = fields["phi"]
    p = fields["p"]
    u = fields["u"]
    v = fields["v"]
    up = fields["up"]
    vp = fields["vp"]

    # ------------------------------------------------------------
    # 3. Apply initial boundary condition
    # ------------------------------------------------------------
    u, v = apply_velocity_bc(u, v, u_lid, nx, ny)
    up, vp = apply_velocity_bc(up, vp, u_lid, nx, ny)

    error_history = []

    # ------------------------------------------------------------
    # 4. Time marching loop
    # ------------------------------------------------------------
    timestep = -1
    interrupted = False

    try:
        for timestep in range(config.max_timestep):

            # --------------------------------------------------------
            # 4-1. Compute momentum RHS
            # --------------------------------------------------------
            rhsu = compute_u_rhs(u, v, up, vp, dx, dy, dt, re)
            rhsv = compute_v_rhs(u, v, up, vp, dx, dy, dt, re)

            # --------------------------------------------------------
            # 4-2. Solve implicit diffusion part using TDMA
            # --------------------------------------------------------
            du = solve_u_diffusion_tdma(rhsu, dx, dy, dt, re)
            dv = solve_v_diffusion_tdma(rhsv, dx, dy, dt, re)

            # --------------------------------------------------------
            # 4-3. Intermediate velocity
            # --------------------------------------------------------
            uhat = u + du
            vhat = v + dv

            uhat, vhat = apply_velocity_bc(uhat, vhat, u_lid, nx, ny)

            # --------------------------------------------------------
            # 4-4. Build pressure Poisson RHS
            # --------------------------------------------------------
            f = compute_divergence(uhat, vhat, dx, dy, dt)

            # --------------------------------------------------------
            # 4-5. Solve pressure correction Poisson equation
            # --------------------------------------------------------
            phi = solve_poisson(f, dx, dy, config, initial=phi)

            # --------------------------------------------------------
            # 4-6. Velocity projection
            # --------------------------------------------------------
            un, vn = project_velocity(uhat, vhat, phi, dx, dy, dt)

            un, vn = apply_velocity_bc(un, vn, u_lid, nx, ny)

            # --------------------------------------------------------
            # 4-7. Save previous velocity and update current velocity
            # --------------------------------------------------------
            up = u.copy()
            vp = v.copy()

            u = un.copy()
            v = vn.copy()

            u, v = apply_velocity_bc(u, v, u_lid, nx, ny)
            up, vp = apply_velocity_bc(up, vp, u_lid, nx, ny)

            # --------------------------------------------------------
            # 4-8. Update pressure
            # --------------------------------------------------------
            p = update_pressure(phi, dx, dy, dt, re)

            # --------------------------------------------------------
            # 4-9. Check convergence
            # --------------------------------------------------------
            total_error = compute_velocity_error(u, v, up, vp)
            error_history.append(total_error)

            if timestep % 100 == 0:
                max_u, max_v = get_max_velocity(u, v)

                print(
                    f"[Step {timestep:7d}] "
                    f"max|u| = {max_u:.6f}, "
                    f"max|v| = {max_v:.6f}, "
                    f"error = {total_error:.6e}"
                )

            interval = config.checkpoint_interval
            if (
                checkpoint_callback is not None
                and interval > 0
                and (timestep + 1) % interval == 0
            ):
                checkpoint_callback(
                    {
                        "u": u,
                        "v": v,
                        "p": p,
                        "phi": phi,
                        "error_history": np.array(error_history),
                        "final_timestep": timestep,
                        "interrupted": False,
                    }
                )

            if timestep > 5 and total_error < config.velocity_tolerance:
                print(f"Converged at step {timestep}.")
                break
    except KeyboardInterrupt:
        interrupted = True
        print(f"\nInterrupted at step {timestep}. Saving current state.")

    # ------------------------------------------------------------
    # 5. Return results
    # ------------------------------------------------------------
    results = {
        "u": u,
        "v": v,
        "p": p,
        "phi": phi,
        "error_history": np.array(error_history),
        "final_timestep": timestep,
        "interrupted": interrupted,
    }

    return results
