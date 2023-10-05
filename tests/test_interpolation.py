import numpy as np
import pytest
from cases import real_case_list, complex_case_list, x, fun_from_expr
from cheby import RealFunction, ComplexFunction

rel_tol = 1.0e-13
abs_tol = 1.0e-13


@pytest.mark.parametrize("case", real_case_list)
def test_interpolation_r(case):
    fun = case[0]
    xmin = case[1]
    xmax = case[2]
    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)
    xn = np.linspace(xmin, xmax, 256)
    f_ex = ff(xn)
    f_num = f(xn)

    error = np.max(np.abs(f_ex - f_num))
    norm = np.max(np.abs(f_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("case", real_case_list + complex_case_list)
def test_interpolation_c(case):
    fun = case[0]
    xmin = case[1]
    xmax = case[2]
    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)
    xn = np.linspace(xmin, xmax, 256)
    f_ex = ff(xn)
    f_num = f(xn)

    error = np.max(np.abs(f_ex - f_num))
    norm = np.max(np.abs(f_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol
