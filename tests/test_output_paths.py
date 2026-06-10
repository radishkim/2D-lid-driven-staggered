from lid_driven.output import (
    format_reynolds_number,
    get_case_name,
    get_data_path,
    get_figure_dir,
)


def test_integer_reynolds_number_has_no_decimal_suffix():
    assert format_reynolds_number(1000.0) == "1000"


def test_case_name_uses_reynolds_number_and_solver():
    assert get_case_name(1000, "SOR") == "Re1000_sor"


def test_output_paths_use_the_same_case_name(tmp_path):
    data_path = get_data_path(tmp_path, 1000, "sor")
    figure_dir = get_figure_dir(tmp_path, 1000, "sor")

    assert data_path == (
        tmp_path / "results" / "data" / "solution_Re1000_sor.npz"
    )
    assert figure_dir == (
        tmp_path / "results" / "figures" / "Re1000_sor"
    )
