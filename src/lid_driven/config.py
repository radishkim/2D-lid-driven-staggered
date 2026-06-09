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

    # Convergence criteria
    velocity_tolerance: float = 1.0e-5
    pressure_tolerance: float = 1.0e-5

    # Lid velocity
    u_lid: float = 1.0

    @property
    def dx(self) -> float:
        return (self.xmax - self.x0) / (self.nx - 2)

    @property
    def dy(self) -> float:
        return (self.ymax - self.y0) / (self.ny - 2)