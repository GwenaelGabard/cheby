import numpy as np
import pytest
from cases import sub_vec, real_case_list, complex_case_list, x, fun_from_expr
from cheby import RealFunction, ComplexFunction

rel_tol = 1.0e-12
abs_tol = 1.0e-12


@pytest.mark.parametrize("case", real_case_list)
def test_real_derivative(case):
    fun = case[0]
    xmin = case[1]
    xmax = case[2]
    dfun = fun.diff()

    ff = fun_from_expr(x, fun)
    dff = fun_from_expr(x, dfun)

    f = RealFunction(ff, xmin, xmax)
    df_ex = RealFunction(dff, xmin, xmax)
    df_num = f.derivative()

    delta = sub_vec(df_ex.coef(), df_num.coef())
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(df_ex.coef()))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("case", real_case_list + complex_case_list)
def test_complex_derivative(case):
    fun = case[0]
    xmin = case[1]
    xmax = case[2]
    dfun = fun.diff()

    ff = fun_from_expr(x, fun)
    dff = fun_from_expr(x, dfun)

    f = ComplexFunction(ff, xmin, xmax)
    df_ex = ComplexFunction(dff, xmin, xmax)
    df_num = f.derivative()

    delta = sub_vec(df_ex.coef(), df_num.coef())
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(df_ex.coef()))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol
