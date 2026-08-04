import numpy as np
import pytest
from cases import complex_case_list, fun_from_expr, real_case_list, sub_vec, x
from cheby import ComplexFunction, RealFunction

rel_tol = 1.0e-12
abs_tol = 1.0e-12


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list)
def test_real_real_part(fun, xmin, xmax):
    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)

    c_ex = f.coef.real
    c_num = f.real().coef

    delta = sub_vec(c_ex, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(c_ex))

    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list)
def test_real_imag_part(fun, xmin, xmax):
    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)

    c_ex = f.coef.imag
    c_num = f.imag().coef

    delta = sub_vec(c_ex, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(c_ex))

    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list)
def test_real_conj(fun, xmin, xmax):
    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)

    c_ex = f.coef.conj()
    c_num = f.conj().coef

    delta = sub_vec(c_ex, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(c_ex))

    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list + complex_case_list)
def test_complex_real_part(fun, xmin, xmax):
    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)

    c_ex = f.coef.real
    c_num = f.real().coef

    delta = sub_vec(c_ex, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(c_ex))

    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list + complex_case_list)
def test_complex_imag_part(fun, xmin, xmax):
    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)

    c_ex = f.coef.imag
    c_num = f.imag().coef

    delta = sub_vec(c_ex, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(c_ex))

    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list + complex_case_list)
def test_complex_conj(fun, xmin, xmax):
    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)

    c_ex = f.coef.conj()
    c_num = f.conj().coef

    delta = sub_vec(c_ex, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(c_ex))

    assert error <= abs_tol + rel_tol * norm
