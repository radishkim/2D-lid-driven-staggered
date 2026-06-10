from pathlib import Path


def format_reynolds_number(reynolds_number):
    """Return a stable Reynolds-number label for output paths."""
    value = float(reynolds_number)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def get_case_name(reynolds_number, poisson_solver):
    """Return the shared case name used by data and figure outputs."""
    re_label = format_reynolds_number(reynolds_number)
    solver_name = poisson_solver.lower()
    return f"Re{re_label}_{solver_name}"


def get_data_path(project_root, reynolds_number, poisson_solver):
    """Return the .npz path for a simulation case."""
    case_name = get_case_name(reynolds_number, poisson_solver)
    return Path(project_root) / "results" / "data" / f"solution_{case_name}.npz"


def get_figure_dir(project_root, reynolds_number, poisson_solver):
    """Return the figure directory for a simulation case."""
    case_name = get_case_name(reynolds_number, poisson_solver)
    return Path(project_root) / "results" / "figures" / case_name
