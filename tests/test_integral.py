import numpy as np
import sympy as sp
import pytest
from cases import complex_case_list, real_case_list, x, fun_from_expr
from cheby import RealFunction, ComplexFunction

rel_tol = 1.0e-12
abs_tol = 1.0e-12
segments = [(-1.0, +1.0), (-1.0, 0.1), (-0.2, +1.0), (-0.21, 0.11), (0.45, 0.45)]


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list)
def test_real_integral(fun, xmin, xmax):
    i_ex = sp.integrate(fun, (x, xmin, xmax)).evalf()

    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)
    i_num = f.integral()

    error = np.abs(i_ex - i_num)
    norm = np.abs(i_ex)

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list)
@pytest.mark.parametrize("a, b", segments)
def test_real_integral2(fun, xmin, xmax, a, b):
    start = (xmax - xmin) * (a + 1) / 2 + xmin
    end = (xmax - xmin) * (b + 1) / 2 + xmin
    i_ex = sp.integrate(fun, (x, start, end)).evalf()

    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)
    i_num = f.integral(start, end)

    error = np.abs(i_ex - i_num)
    norm = np.abs(i_ex)

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list + complex_case_list)
def test_complex_integral(fun, xmin, xmax):
    i_ex = sp.integrate(fun, (x, xmin, xmax)).evalf()

    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)
    i_num = f.integral()

    error = np.abs(i_ex - i_num)
    norm = np.abs(i_ex)

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list + complex_case_list)
@pytest.mark.parametrize("a, b", segments)
def test_complex_integral2(fun, xmin, xmax, a, b):
    start = (xmax - xmin) * (a + 1) / 2 + xmin
    end = (xmax - xmin) * (b + 1) / 2 + xmin
    i_ex = sp.integrate(fun, (x, start, end)).evalf()

    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)
    i_num = f.integral(start, end)

    error = np.abs(i_ex - i_num)
    norm = np.abs(i_ex)

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol
