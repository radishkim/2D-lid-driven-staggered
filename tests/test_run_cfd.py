import pytest

from scripts.run_cfd import (
    POISSON_SOLVER,
    REYNOLDS_NUMBER,
    parse_args,
)


def test_parse_args_uses_run_script_defaults():
    args = parse_args([])

    assert args.poisson_solver == POISSON_SOLVER
    assert args.re == REYNOLDS_NUMBER


def test_parse_args_accepts_solver_and_reynolds_number():
    args = parse_args(["multigrid", "--re", "500"])

    assert args.poisson_solver == "multigrid"
    assert args.re == 500.0


def test_parse_args_rejects_nonpositive_reynolds_number():
    with pytest.raises(SystemExit):
        parse_args(["--re", "0"])
