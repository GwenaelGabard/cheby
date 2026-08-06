import numpy as np
import pytest
from cases import (
    fun_from_expr,
    prod_list_cc,
    prod_list_cr,
    prod_list_rc,
    prod_list_rr,
    sub_vec,
    x,
)
from cheby import ComplexFunction, RealFunction

rel_tol = 5.0e-12
abs_tol = 5.0e-12
# For functions with large Chebyshev coefficients, round-off error in the product can be
# much larger than the final result, so we adjust the tolerance by adding a term
# proportional to the product of the two operands' coefficient magnitude.
cond_tol = 100 * np.finfo(float).eps


@pytest.mark.parametrize("fun1, fun2, xmin, xmax", prod_list_rr)
def test_multiply_rr(fun1, fun2, xmin, xmax):
    prod = fun1 * fun2

    ff1 = fun_from_expr(x, fun1)
    ff2 = fun_from_expr(x, fun2)
    fp = fun_from_expr(x, prod)

    f1 = RealFunction(ff1, xmin, xmax)
    f2 = RealFunction(ff2, xmin, xmax)
    p_ex = RealFunction(fp, xmin, xmax)
    p_num = f1 * f2

    delta = sub_vec(p_ex.coef, p_num.coef)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(p_ex.coef))

    scale = np.max(np.abs(f1.coef)) * np.max(np.abs(f2.coef))
    assert error <= abs_tol + rel_tol * norm + cond_tol * scale


@pytest.mark.parametrize("fun1, fun2, xmin, xmax", prod_list_cr)
def test_multiply_cr(fun1, fun2, xmin, xmax):
    prod = fun1 * fun2

    ff1 = fun_from_expr(x, fun1)
    ff2 = fun_from_expr(x, fun2)
    fp = fun_from_expr(x, prod)

    f1 = ComplexFunction(ff1, xmin, xmax)
    f2 = RealFunction(ff2, xmin, xmax)
    p_ex = ComplexFunction(fp, xmin, xmax)
    p_num = f1 * f2

    delta = sub_vec(p_ex.coef, p_num.coef)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(p_ex.coef))

    scale = np.max(np.abs(f1.coef)) * np.max(np.abs(f2.coef))
    assert error <= abs_tol + rel_tol * norm + cond_tol * scale


@pytest.mark.parametrize("fun1, fun2, xmin, xmax", prod_list_rc)
def test_multiply_rc(fun1, fun2, xmin, xmax):
    prod = fun1 * fun2

    ff1 = fun_from_expr(x, fun1)
    ff2 = fun_from_expr(x, fun2)
    fp = fun_from_expr(x, prod)

    f1 = RealFunction(ff1, xmin, xmax)
    f2 = ComplexFunction(ff2, xmin, xmax)
    p_ex = ComplexFunction(fp, xmin, xmax)
    p_num = f1 * f2

    delta = sub_vec(p_ex.coef, p_num.coef)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(p_ex.coef))

    scale = np.max(np.abs(f1.coef)) * np.max(np.abs(f2.coef))
    assert error <= abs_tol + rel_tol * norm + cond_tol * scale


@pytest.mark.parametrize("fun1, fun2, xmin, xmax", prod_list_cc)
def test_multiply_cc(fun1, fun2, xmin, xmax):
    prod = fun1 * fun2

    ff1 = fun_from_expr(x, fun1)
    ff2 = fun_from_expr(x, fun2)
    fp = fun_from_expr(x, prod)

    f1 = ComplexFunction(ff1, xmin, xmax)
    f2 = ComplexFunction(ff2, xmin, xmax)
    p_ex = ComplexFunction(fp, xmin, xmax)
    p_num = f1 * f2

    delta = sub_vec(p_ex.coef, p_num.coef)
    error = np.max(np.abs(delta))
    norm = np.max(np.abs(p_ex.coef))

    scale = np.max(np.abs(f1.coef)) * np.max(np.abs(f2.coef))
    assert error <= abs_tol + rel_tol * norm + cond_tol * scale
