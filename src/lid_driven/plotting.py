import os

import matplotlib.pyplot as plt
import numpy as np


def _prepare_output_dir(save_dir):
    """
    Create output directory if it does not exist.
    """
    os.makedirs(save_dir, exist_ok=True)


def plot_velocity_magnitude(x, y, velocity_magnitude, save_dir="results/figures"):
    """
    Plot velocity magnitude contour.
    """
    _prepare_output_dir(save_dir)

    X, Y = np.meshgrid(x, y, indexing="ij")

    plt.figure(figsize=(6, 5))

    levels = np.linspace(
        np.min(velocity_magnitude),
        np.max(velocity_magnitude),
        30,
    )

    contour = plt.contour(
        X,
        Y,
        velocity_magnitude,
        levels=levels,
        linewidths=0.8,
    )

    plt.clabel(contour, inline=True, fontsize=8)
    plt.colorbar(contour, label="Velocity magnitude")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Velocity Magnitude")
    plt.gca().set_aspect("equal")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "velocity_magnitude_Re100.png"), dpi=300)
    plt.show()


def plot_pressure(x, y, pcol, save_dir="results/figures"):
    """
    Plot pressure contour.
    """
    _prepare_output_dir(save_dir)

    X, Y = np.meshgrid(x, y, indexing="ij")

    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, pcol, levels=50)
    plt.colorbar(contour, label="Pressure")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Pressure Contour")
    plt.gca().set_aspect("equal")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pressure_Re100.png"), dpi=300)
    plt.show()


def plot_vorticity(x, y, omega, save_dir="results/figures"):
    """
    Plot vorticity contour lines.
    """
    _prepare_output_dir(save_dir)

    X, Y = np.meshgrid(x, y, indexing="ij")

    plt.figure(figsize=(6, 5))

    levels = np.linspace(
        np.min(omega),
        np.max(omega),
        40,
    )

    contour = plt.contour(
        X,
        Y,
        omega,
        levels=levels,
        linewidths=0.8,
    )

    plt.clabel(contour, inline=True, fontsize=8)
    plt.colorbar(contour, label="Vorticity")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Vorticity Contour Lines")
    plt.gca().set_aspect("equal")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vorticity_contour_Re100.png"), dpi=300)
    plt.show()


def plot_streamlines(x, y, ucol, vcol, save_dir="results/figures"):
    """
    Plot streamlines from collocated velocity fields.
    """
    _prepare_output_dir(save_dir)

    X, Y = np.meshgrid(x, y, indexing="ij")

    plt.figure(figsize=(6, 5))

    # matplotlib streamplot expects arrays as [y, x] layout,
    # so transpose X, Y, u, v for plotting.
    plt.streamplot(
        X.T,
        Y.T,
        ucol.T,
        vcol.T,
        density=1.5,
        linewidth=1.0,
        arrowsize=1.0,
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Streamlines")
    plt.gca().set_aspect("equal")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "streamlines_Re100.png"), dpi=300)
    plt.show()


def plot_convergence(error_history, save_dir="results/figures"):
    """
    Plot convergence history.
    """
    _prepare_output_dir(save_dir)

    plt.figure(figsize=(6, 4))
    plt.semilogy(error_history)

    plt.xlabel("Time step")
    plt.ylabel("Velocity error")
    plt.title("Convergence History")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "convergence_Re100.png"), dpi=300)
    plt.show()