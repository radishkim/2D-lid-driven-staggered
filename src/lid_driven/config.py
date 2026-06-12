from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """
    Configuration for the 2D lid-driven cavity simulation.
    """

    # Domain size
    x0: float = 0.0
    xmax: float = 1.0
    y0: float = 0.0
    ymax: float = 1.0

    # Grid size
    nx: int = 130
    ny: int = 130

    # Physical and numerical parameters
    re: float = 100.0
    dt: float = 0.005
    max_timestep: int = 1_000_000
    checkpoint_interval: int = 1000

    # Convergence criteria
    velocity_tolerance: float = 1.0e-5
    pressure_tolerance: float = 1.0e-5

    # Lid velocity
    u_lid: float = 1.0
    
    # Poisson solver option, default is "sor"
    poisson_solver: str = "sor"

    # SOR parameters
    sor_omega: float = 1.7
    sor_tolerance: float = 1.0e-5
    sor_max_iter: int = 1000

    # Multigrid parameters
    multigrid_tolerance: float = 1.0e-5
    multigrid_max_cycles: int = 100
    multigrid_pre_smooth: int = 3
    multigrid_post_smooth: int = 3
    multigrid_omega: float = 0.8

    # BiCGSTAB parameters
    bicgstab_tolerance: float = 1.0e-5
    bicgstab_max_iter: int = 10000

    @property
    def dx(self) -> float:
        return (self.xmax - self.x0) / (self.nx - 2)

    @property
    def dy(self) -> float:
        return (self.ymax - self.y0) / (self.ny - 2)
