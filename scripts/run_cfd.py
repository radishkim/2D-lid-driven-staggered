import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Add src directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)

from lid_driven.config import SimulationConfig
from lid_driven.grid import create_grid
from lid_driven.output import get_data_path
from lid_driven.solver import run_simulation


# These defaults are used when the corresponding command-line arguments
# are omitted. Command-line values take precedence over these settings.
REYNOLDS_NUMBER = 100.0
POISSON_SOLVER = "sor"


def positive_float(value):
    value = float(value)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("Reynolds number must be positive.")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the 2D lid-driven cavity simulation."
    )
    parser.add_argument(
        "poisson_solver",
        nargs="?",
        default=POISSON_SOLVER,
        choices=("dct", "sor", "multigrid", "bicgstab"),
        help=f"pressure Poisson solver (default: {POISSON_SOLVER})",
    )
    parser.add_argument(
        "--re",
        type=positive_float,
        default=REYNOLDS_NUMBER,
        metavar="VALUE",
        help=f"Reynolds number (default: {REYNOLDS_NUMBER:g})",
    )
    return parser.parse_args(argv)


def save_results(save_path, results, config, x, y):
    solver_name = config.poisson_solver.lower()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = save_path.with_suffix(".tmp.npz")

    np.savez(
        temporary_path,
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
        poisson_solver=solver_name,
        interrupted=results.get("interrupted", False),
    )
    os.replace(temporary_path, save_path)


def main():
    args = parse_args()
    # Runtime case selection belongs to this entry point; SimulationConfig
    # supplies the remaining numerical and grid settings.
    config = SimulationConfig(
        re=args.re,
        poisson_solver=args.poisson_solver,
    )

    x, y = create_grid(config)
    solver_name = config.poisson_solver.lower()
    save_path = get_data_path(PROJECT_ROOT, config.re, solver_name)

    def save_checkpoint(results):
        save_results(save_path, results, config, x, y)
        print(
            f"Checkpoint saved at step {results['final_timestep']}: "
            f"{save_path}"
        )

    results = run_simulation(
        config,
        checkpoint_callback=save_checkpoint,
    )
    save_results(save_path, results, config, x, y)

    print("Simulation finished.")
    print(f"Final timestep: {results['final_timestep']}")
    print(f"Interrupted: {results['interrupted']}")
    print(f"u shape: {results['u'].shape}")
    print(f"v shape: {results['v'].shape}")
    print(f"p shape: {results['p'].shape}")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"Solution saved to {save_path}")


if __name__ == "__main__":
    main()
