import numpy as np
import pytest
from cases import complex_root_cases, fun_from_expr, real_root_cases, x
from cheby import ComplexFunction, RealFunction

rel_tol = 1.0e-9
abs_tol = 1.0e-9


@pytest.mark.parametrize("fun, xmin, xmax, expected", real_root_cases)
def test_roots_real(fun, xmin, xmax, expected):
    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)
    roots = np.sort(f.roots())

    assert len(roots) == len(expected)
    error = np.max(np.abs(roots - np.array(expected)))
    assert error <= abs_tol + rel_tol * np.max(np.abs(expected))


@pytest.mark.parametrize(
    "fun, xmin, xmax, expected", real_root_cases + complex_root_cases
)
def test_roots_complex(fun, xmin, xmax, expected):
    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)
    roots = np.sort(f.roots())

    assert len(roots) == len(expected)
    error = np.max(np.abs(roots - np.array(expected)))
    assert error <= abs_tol + rel_tol * np.max(np.abs(expected))
