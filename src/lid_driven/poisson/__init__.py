from .base import solve_poisson
from .bicgstab import solve_poisson_bicgstab
from .dct import solve_poisson_dct
from .multigrid import solve_poisson_multigrid
from .sor import solve_poisson_sor

__all__ = [
    "solve_poisson",
    "solve_poisson_bicgstab",
    "solve_poisson_dct",
    "solve_poisson_multigrid",
    "solve_poisson_sor",
]
