import numpy as np
import pytest
from cases import sub_vec, complex_case_list, real_case_list, x, fun_from_expr
from cheby import RealFunction, ComplexFunction

rel_tol = 1.0e-12
abs_tol = 1.0e-12


@pytest.mark.parametrize("case", real_case_list)
def test_real_real_part(case):
    fun, xmin, xmax = case

    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)

    c_ex = f.coef().real
    c_num = f.real().coef()

    delta = sub_vec(c_ex, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(c_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("case", real_case_list)
def test_real_imag_part(case):
    fun, xmin, xmax = case

    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)

    c_ex = f.coef().imag
    c_num = f.imag().coef()

    delta = sub_vec(c_ex, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(c_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("case", real_case_list)
def test_real_conj(case):
    fun, xmin, xmax = case

    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)

    c_ex = f.coef().conj()
    c_num = f.conj().coef()

    delta = sub_vec(c_ex, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(c_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("case", real_case_list + complex_case_list)
def test_complex_real_part(case):
    fun, xmin, xmax = case

    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)

    c_ex = f.coef().real
    c_num = f.real().coef()

    delta = sub_vec(c_ex, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(c_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("case", real_case_list + complex_case_list)
def test_complex_imag_part(case):
    fun, xmin, xmax = case

    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)

    c_ex = f.coef().imag
    c_num = f.imag().coef()

    delta = sub_vec(c_ex, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(c_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("case", real_case_list + complex_case_list)
def test_complex_conj(case):
    fun, xmin, xmax = case

    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)

    c_ex = f.coef().conj()
    c_num = f.conj().coef()

    delta = sub_vec(c_ex, c_num)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(c_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol
