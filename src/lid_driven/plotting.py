import os

import matplotlib.pyplot as plt
import numpy as np


def _prepare_output_dir(save_dir):
    """
    Create output directory if it does not exist.
    """
    os.makedirs(save_dir, exist_ok=True)


def _figure_filename(name, case_name):
    suffix = f"_{case_name}" if case_name else ""
    return f"{name}{suffix}.png"


def plot_velocity_magnitude(
    x,
    y,
    velocity_magnitude,
    save_dir="results/figures",
    case_name=None,
):
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
    filename = _figure_filename("velocity_magnitude", case_name)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()


def plot_pressure(
    x,
    y,
    pcol,
    save_dir="results/figures",
    case_name=None,
):
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
    filename = _figure_filename("pressure", case_name)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()


def plot_vorticity(
    x,
    y,
    omega,
    save_dir="results/figures",
    case_name=None,
):
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
    filename = _figure_filename("vorticity_contour", case_name)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()


def plot_streamlines(
    x,
    y,
    ucol,
    vcol,
    save_dir="results/figures",
    case_name=None,
):
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
    filename = _figure_filename("streamlines", case_name)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()


def plot_convergence(
    error_history,
    save_dir="results/figures",
    case_name=None,
):
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
    filename = _figure_filename("convergence", case_name)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()
