import numpy as np
import pytest
from cases import complex_root_cases, fun_from_expr, real_root_cases, x
from cheby import ComplexFunction, RealFunction

rel_tol = 1.0e-6
abs_tol = 1.0e-6
imag_tol = 1.0e-7


def eigen_roots(matrix, xmin, xmax):
    values = np.linalg.eigvals(matrix)
    mask = (
        (np.abs(values.imag) <= imag_tol)
        & (values.real >= -1.0 - 1e-9)
        & (values.real <= 1.0 + 1e-9)
    )
    xi = np.sort(values.real[mask])
    return (xi + 1.0) / 2.0 * (xmax - xmin) + xmin


@pytest.mark.parametrize("fun, xmin, xmax, expected", real_root_cases)
def test_colleague_real(fun, xmin, xmax, expected):
    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)

    roots = eigen_roots(f.colleague(), xmin, xmax)

    assert len(roots) == len(expected)
    error = np.max(np.abs(roots - np.array(expected)))
    assert error <= abs_tol + rel_tol * np.max(np.abs(expected))


@pytest.mark.parametrize(
    "fun, xmin, xmax, expected", real_root_cases + complex_root_cases
)
def test_colleague_complex(fun, xmin, xmax, expected):
    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)

    roots = eigen_roots(f.colleague(), xmin, xmax)

    assert len(roots) == len(expected)
    error = np.max(np.abs(roots - np.array(expected)))
    assert error <= abs_tol + rel_tol * np.max(np.abs(expected))
