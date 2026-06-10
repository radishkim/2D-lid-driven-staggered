import os
import sys
import numpy as np

# Add src directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)

from lid_driven.config import SimulationConfig
from lid_driven.grid import create_grid
from lid_driven.solver import run_simulation


def main():
    config = SimulationConfig()

    x, y = create_grid(config)

    results = run_simulation(config)

    os.makedirs("results/data", exist_ok=True)

    save_path = "results/data/solution_Re100_dct.npz"

    np.savez(
        save_path,
        u=results["u"],
        v=results["v"],
        p=results["p"],
        phi=results["phi"],
        error_history=results["error_history"],
        final_timestep=results["final_timestep"],
        x=x,
        y=y,
        nx=config.nx,
        ny=config.ny,
        dx=config.dx,
        dy=config.dy,
        re=config.re,
        dt=config.dt,
    )

    print("Simulation finished.")
    print(f"Final timestep: {results['final_timestep']}")
    print(f"u shape: {results['u'].shape}")
    print(f"v shape: {results['v'].shape}")
    print(f"p shape: {results['p'].shape}")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"Solution saved to {save_path}")


if __name__ == "__main__":
    main()