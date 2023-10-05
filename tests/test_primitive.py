import numpy as np
import sympy as sp
import pytest
from cases import sub_vec, complex_case_list, real_case_list, x, fun_from_expr
from cheby import RealFunction, ComplexFunction

rel_tol = 1.0e-12
abs_tol = 1.0e-12


@pytest.mark.parametrize("case", real_case_list)
def test_real_primitive(case):
    fun = case[0]
    xmin = case[1]
    xmax = case[2]
    ifun = sp.integrate(fun, x)

    ff = fun_from_expr(x, fun)
    iff = fun_from_expr(x, ifun)

    f = RealFunction(ff, xmin, xmax)
    if_ex = RealFunction(iff, xmin, xmax)
    if_num = f.primitive()

    delta = sub_vec(if_ex.coef(), if_num.coef())
    delta[0] = 0.0
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(if_ex.coef()))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("case", real_case_list + complex_case_list)
def test_complex_primitive(case):
    fun = case[0]
    xmin = case[1]
    xmax = case[2]
    ifun = sp.integrate(fun, x)

    ff = fun_from_expr(x, fun)
    iff = fun_from_expr(x, ifun)

    f = ComplexFunction(ff, xmin, xmax)
    if_ex = ComplexFunction(iff, xmin, xmax)
    if_num = f.primitive()

    delta = sub_vec(if_ex.coef(), if_num.coef())
    delta[0] = 0.0
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(if_ex.coef()))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol
