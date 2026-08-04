import numpy as np
import pytest
import sympy as sp
from cases import complex_case_list, fun_from_expr, real_case_list, sub_vec, x
from cheby import ComplexFunction, RealFunction

rel_tol = 1.0e-12
abs_tol = 1.0e-12


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list)
def test_real_primitive(fun, xmin, xmax):
    ifun = sp.integrate(fun, x)

    ff = fun_from_expr(x, fun)
    iff = fun_from_expr(x, ifun)

    f = RealFunction(ff, xmin, xmax)
    if_ex = RealFunction(iff, xmin, xmax)
    if_num = f.primitive()

    delta = sub_vec(if_ex.coef, if_num.coef)
    delta[0] = 0.0
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(if_ex.coef))

    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list + complex_case_list)
def test_complex_primitive(fun, xmin, xmax):
    ifun = sp.integrate(fun, x)

    ff = fun_from_expr(x, fun)
    iff = fun_from_expr(x, ifun)

    f = ComplexFunction(ff, xmin, xmax)
    if_ex = ComplexFunction(iff, xmin, xmax)
    if_num = f.primitive()

    delta = sub_vec(if_ex.coef, if_num.coef)
    delta[0] = 0.0
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(if_ex.coef))

    assert error <= abs_tol + rel_tol * norm
