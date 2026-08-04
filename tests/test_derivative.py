import numpy as np
import pytest
from cases import complex_case_list, fun_from_expr, real_case_list, sub_vec, x
from cheby import ComplexFunction, RealFunction

rel_tol = 1.0e-12
abs_tol = 1.0e-12


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list)
def test_real_derivative(fun, xmin, xmax):
    dfun = fun.diff()

    ff = fun_from_expr(x, fun)
    dff = fun_from_expr(x, dfun)

    f = RealFunction(ff, xmin, xmax)
    df_ex = RealFunction(dff, xmin, xmax)
    df_num = f.derivative()

    delta = sub_vec(df_ex.coef, df_num.coef)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(df_ex.coef))

    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list + complex_case_list)
def test_complex_derivative(fun, xmin, xmax):
    dfun = fun.diff()

    ff = fun_from_expr(x, fun)
    dff = fun_from_expr(x, dfun)

    f = ComplexFunction(ff, xmin, xmax)
    df_ex = ComplexFunction(dff, xmin, xmax)
    df_num = f.derivative()

    delta = sub_vec(df_ex.coef, df_num.coef)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(df_ex.coef))

    assert error <= abs_tol + rel_tol * norm
