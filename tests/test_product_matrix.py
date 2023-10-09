import numpy as np
import pytest
from cases import (
    sub_vec,
    prod_list_rr,
    prod_list_cr,
    prod_list_rc,
    prod_list_cc,
    x,
    fun_from_expr,
)
from cheby import RealFunction, ComplexFunction

rel_tol = 1.0e-12
abs_tol = 1.0e-12


@pytest.mark.parametrize("case", prod_list_rr)
def test_matprod_rr(case):
    fun1 = case[0]
    fun2 = case[1]
    xmin = case[2]
    xmax = case[3]
    prod = fun1 * fun2

    ff1 = fun_from_expr(x, fun1)
    ff2 = fun_from_expr(x, fun2)
    fp = fun_from_expr(x, prod)

    f1 = RealFunction(ff1, xmin, xmax)
    f2 = RealFunction(ff2, xmin, xmax)
    p_ex = RealFunction(fp, xmin, xmax)
    c_num = f1.product_matrix(len(f2.coef()) - 1) @ f2.coef()[:, None]

    delta = sub_vec(p_ex.coef(), c_num.flatten())
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(p_ex.coef()))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("case", prod_list_cr)
def test_matprod_cr(case):
    fun1 = case[0]
    fun2 = case[1]
    xmin = case[2]
    xmax = case[3]
    prod = fun1 * fun2

    ff1 = fun_from_expr(x, fun1)
    ff2 = fun_from_expr(x, fun2)
    fp = fun_from_expr(x, prod)

    f1 = ComplexFunction(ff1, xmin, xmax)
    f2 = RealFunction(ff2, xmin, xmax)
    p_ex = ComplexFunction(fp, xmin, xmax)
    c_num = f1.product_matrix(len(f2.coef()) - 1) @ f2.coef()[:, None]

    delta = sub_vec(p_ex.coef(), c_num.flatten())
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(p_ex.coef()))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("case", prod_list_rc)
def test_matprod_rc(case):
    fun1 = case[0]
    fun2 = case[1]
    xmin = case[2]
    xmax = case[3]
    prod = fun1 * fun2

    ff1 = fun_from_expr(x, fun1)
    ff2 = fun_from_expr(x, fun2)
    fp = fun_from_expr(x, prod)

    f1 = RealFunction(ff1, xmin, xmax)
    f2 = ComplexFunction(ff2, xmin, xmax)
    p_ex = ComplexFunction(fp, xmin, xmax)
    c_num = f1.product_matrix(len(f2.coef()) - 1) @ f2.coef()[:, None]

    delta = sub_vec(p_ex.coef(), c_num.flatten())
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(p_ex.coef()))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("case", prod_list_cc)
def test_matprod_cc(case):
    fun1 = case[0]
    fun2 = case[1]
    xmin = case[2]
    xmax = case[3]
    prod = fun1 * fun2

    ff1 = fun_from_expr(x, fun1)
    ff2 = fun_from_expr(x, fun2)
    fp = fun_from_expr(x, prod)

    f1 = ComplexFunction(ff1, xmin, xmax)
    f2 = ComplexFunction(ff2, xmin, xmax)
    p_ex = ComplexFunction(fp, xmin, xmax)
    c_num = f1.product_matrix(len(f2.coef()) - 1) @ f2.coef()[:, None]

    delta = sub_vec(p_ex.coef(), c_num.flatten())
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(p_ex.coef()))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol
