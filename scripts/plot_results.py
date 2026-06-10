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
from lid_driven.plotting import (
    plot_velocity_magnitude,
    plot_pressure,
    plot_vorticity,
    plot_streamlines,
    plot_convergence,
)


def main():
    data_path = "results/data/solution_Re100_dct.npz"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} does not exist.\n"
            "First run: python scripts/run_re100.py"
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

    plot_velocity_magnitude(x, y, velocity_magnitude)
    plot_pressure(x, y, pcol)
    plot_vorticity(x, y, omega)
    plot_streamlines(x, y, ucol, vcol)
    plot_convergence(error_history)

    print("Visualization finished.")
    print(f"Re = {re}")
    print(f"u shape: {u.shape}")
    print(f"v shape: {v.shape}")
    print(f"p shape: {p.shape}")
    print(f"ucol shape: {ucol.shape}")
    print(f"vcol shape: {vcol.shape}")
    print(f"pcol shape: {pcol.shape}")
    print("Figures saved in results/figures/")


if __name__ == "__main__":
    main()