import argparse
import os
import sys
import numpy as np

# Add src directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)

from lid_driven.postprocess import (
    staggered_to_collocated,
    compute_vorticity,
    compute_velocity_magnitude,
)
from lid_driven.config import SimulationConfig
from lid_driven.output import get_case_name, get_data_path, get_figure_dir
from lid_driven.plotting import (
    plot_velocity_magnitude,
    plot_pressure,
    plot_vorticity,
    plot_streamlines,
    plot_convergence,
)
from run_cfd import POISSON_SOLVER


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "poisson_solver",
        nargs="?",
        default=POISSON_SOLVER,
        choices=("dct", "sor", "multigrid", "bicgstab"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = SimulationConfig(poisson_solver=args.poisson_solver)
    solver_name = config.poisson_solver.lower()
    case_name = get_case_name(config.re, solver_name)
    data_path = get_data_path(PROJECT_ROOT, config.re, solver_name)
    figure_dir = get_figure_dir(PROJECT_ROOT, config.re, solver_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} does not exist.\n"
            f"First run: python scripts/run_cfd.py {solver_name}"
        )

    data = np.load(data_path)

    u = data["u"]
    v = data["v"]
    p = data["p"]
    x = data["x"]
    y = data["y"]
    error_history = data["error_history"]

    nx = int(data["nx"])
    ny = int(data["ny"])
    dx = float(data["dx"])
    dy = float(data["dy"])
    re = float(data["re"])

    ucol, vcol, pcol = staggered_to_collocated(
        u,
        v,
        p,
        nx,
        ny,
    )

    omega = compute_vorticity(ucol, vcol, dx, dy)
    velocity_magnitude = compute_velocity_magnitude(ucol, vcol)

    plot_velocity_magnitude(
        x,
        y,
        velocity_magnitude,
        save_dir=figure_dir,
        case_name=case_name,
    )
    plot_pressure(
        x, y, pcol, save_dir=figure_dir, case_name=case_name
    )
    plot_vorticity(
        x, y, omega, save_dir=figure_dir, case_name=case_name
    )
    plot_streamlines(
        x,
        y,
        ucol,
        vcol,
        save_dir=figure_dir,
        case_name=case_name,
    )
    plot_convergence(
        error_history, save_dir=figure_dir, case_name=case_name
    )

    print("Visualization finished.")
    print(f"Re = {re}")
    print(f"u shape: {u.shape}")
    print(f"v shape: {v.shape}")
    print(f"p shape: {p.shape}")
    print(f"ucol shape: {ucol.shape}")
    print(f"vcol shape: {vcol.shape}")
    print(f"pcol shape: {pcol.shape}")
    print(f"Figures saved in {figure_dir}/")


if __name__ == "__main__":
    main()
