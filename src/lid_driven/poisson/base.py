from .bicgstab import solve_poisson_bicgstab
from .dct import solve_poisson_dct
from .multigrid import solve_poisson_multigrid
from .sor import solve_poisson_sor


def solve_poisson(rhs, dx, dy, config, initial=None):
    """Dispatch to the Poisson solver selected in ``config``."""
    solver_name = config.poisson_solver.lower()

    if solver_name == "dct":
        return solve_poisson_dct(rhs, dx, dy)

    if solver_name == "sor":
        return solve_poisson_sor(
            rhs,
            dx,
            dy,
            config.sor_omega,
            config.sor_tolerance,
            config.sor_max_iter,
            initial=initial,
        )

    if solver_name == "multigrid":
        return solve_poisson_multigrid(
            rhs,
            dx,
            dy,
            config.multigrid_tolerance,
            config.multigrid_max_cycles,
            config.multigrid_pre_smooth,
            config.multigrid_post_smooth,
            config.multigrid_omega,
            initial=initial,
        )

    if solver_name == "bicgstab":
        return solve_poisson_bicgstab(
            rhs,
            dx,
            dy,
            config.bicgstab_tolerance,
            config.bicgstab_max_iter,
            initial=initial,
        )

    available = "dct, sor, multigrid, bicgstab"
    raise ValueError(
        f"Unknown Poisson solver: {config.poisson_solver!r}. "
        f"Available solvers: {available}."
    )
