import numpy as np
import pytest
from cases import sub_vec, real_case_list, complex_case_list, x, fun_from_expr
from cheby import RealFunction, ComplexFunction

rel_tol = 1.0e-12
abs_tol = 1.0e-12
order_list = [0, 1, 2, 3, 4, 5, 11, 12]


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list)
@pytest.mark.parametrize("order", order_list)
def test_power_real(fun, xmin, xmax, order):
    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax).pow(order)

    ffn = fun_from_expr(x, fun**order)
    f_ex = RealFunction(ffn, xmin, xmax)

    delta = sub_vec(f_ex.coef, f.coef)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(f_ex.coef))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list + complex_case_list)
@pytest.mark.parametrize("order", order_list)
def test_power_complex(fun, xmin, xmax, order):
    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax).pow(order)

    ffn = fun_from_expr(x, fun**order)
    f_ex = ComplexFunction(ffn, xmin, xmax)

    delta = sub_vec(f_ex.coef, f.coef)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(f_ex.coef))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol
