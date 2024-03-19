import numpy as np
import pytest
from cases import sub_vec, real_case_list, complex_case_list, x, fun_from_expr
from cheby import RealFunction, ComplexFunction, Basis1D

rel_tol = 1.0e-12
abs_tol = 1.0e-12


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list)
def test_real_diffmat(fun, xmin, xmax):
    dfun = fun.diff()

    ff = fun_from_expr(x, fun)
    dff = fun_from_expr(x, dfun)

    f = RealFunction(ff, xmin, xmax)
    df_ex = RealFunction(dff, xmin, xmax)
    basis = Basis1D(len(f.coef) - 1, xmin, xmax)
    c_num = basis.diff_matrix() @ f.coef

    delta = sub_vec(df_ex.coef, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(df_ex.coef))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list + complex_case_list)
def test_complex_diffmat(fun, xmin, xmax):
    dfun = fun.diff()

    ff = fun_from_expr(x, fun)
    dff = fun_from_expr(x, dfun)

    f = ComplexFunction(ff, xmin, xmax)
    df_ex = ComplexFunction(dff, xmin, xmax)
    basis = Basis1D(len(f.coef) - 1, xmin, xmax)
    c_num = basis.diff_matrix() @ f.coef

    delta = sub_vec(df_ex.coef, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(df_ex.coef))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol
