import pytest

from nero.kinematics.debug_tools import DEFAULT_NERO_JOINT_LIMITS
from nero.kinematics.solver_debug_adapter import make_debug_solver


def pytest_addoption(parser):
    parser.addoption(
        "--solver",
        action="store",
        default="pinocchio",
        choices=("pinocchio", "original"),
        help="Kinematics solver under test: pinocchio or original.",
    )


@pytest.fixture
def solver_name(request):
    return request.config.getoption("--solver")


@pytest.fixture
def solver_factory(solver_name):
    if solver_name == "pinocchio":
        pytest.importorskip("pinocchio")

    def _make_solver(max_iterations=120, n_psi=181):
        return make_debug_solver(
            solver_name,
            joint_limits=DEFAULT_NERO_JOINT_LIMITS,
            dt=0.05,
            n_psi=n_psi,
            max_iterations=max_iterations,
            tol_pos=1e-5,
            tol_rot=1e-4,
        )

    return _make_solver
