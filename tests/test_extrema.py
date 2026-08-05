import numpy as np
import pytest
from cases import complex_extrema_cases, fun_from_expr, real_extrema_cases, x
from cheby import ComplexFunction, RealFunction

rel_tol = 1.0e-8
abs_tol = 1.0e-8


@pytest.mark.parametrize("fun, xmin, xmax, expected", real_extrema_cases)
def test_extrema_real(fun, xmin, xmax, expected):
    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)
    extrema = np.sort(f.extrema())

    assert len(extrema) == len(expected)
    error = np.max(np.abs(extrema - np.array(expected)))
    assert error <= abs_tol + rel_tol * np.max(np.abs(expected))


@pytest.mark.parametrize(
    "fun, xmin, xmax, expected", real_extrema_cases + complex_extrema_cases
)
def test_extrema_complex(fun, xmin, xmax, expected):
    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)
    extrema = np.sort(f.extrema())

    assert len(extrema) == len(expected)
    error = np.max(np.abs(extrema - np.array(expected)))
    assert error <= abs_tol + rel_tol * np.max(np.abs(expected))
