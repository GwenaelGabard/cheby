import numpy as np
import pytest
from cases import (
    fun_from_expr,
    prod_list_cc,
    prod_list_cr,
    prod_list_rc,
    prod_list_rr,
    x,
)
from cheby import Basis1D, ComplexFunction, RealFunction

rel_tol = 5.0e-12
abs_tol = 5.0e-12


@pytest.mark.parametrize("fun1, fun2, xmin, xmax", prod_list_rr)
def test_basis_projection_rr(fun1, fun2, xmin, xmax):
    prod = fun1 * fun2

    ff1 = fun_from_expr(x, fun1)
    ff2 = fun_from_expr(x, fun2)
    fp = fun_from_expr(x, prod)

    f1 = RealFunction(ff1, xmin, xmax)
    f2 = RealFunction(ff2, xmin, xmax)
    p_ex = RealFunction(fp, xmin, xmax)

    i_ex = p_ex.integral()

    num_coef = max(len(f1.coef), len(f2.coef))
    basis = Basis1D(num_coef - 1, xmin, xmax)
    P = basis.projection_matrix()
    print(f1.coef.shape, P.shape, f2.coef.shape)
    i_num = f1.coef.T @ P[: len(f1.coef), : len(f2.coef)] @ f2.coef

    error = np.abs(i_num - i_ex)
    norm = np.abs(i_ex)

    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("fun1, fun2, xmin, xmax", prod_list_cr)
def test_basis_projection_cr(fun1, fun2, xmin, xmax):
    prod = fun1 * fun2

    ff1 = fun_from_expr(x, fun1)
    ff2 = fun_from_expr(x, fun2)
    fp = fun_from_expr(x, prod)

    f1 = ComplexFunction(ff1, xmin, xmax)
    f2 = RealFunction(ff2, xmin, xmax)
    p_ex = ComplexFunction(fp, xmin, xmax)

    i_ex = p_ex.integral()

    num_coef = max(len(f1.coef), len(f2.coef))
    basis = Basis1D(num_coef - 1, xmin, xmax)
    P = basis.projection_matrix()
    i_num = f1.coef.T @ P[: len(f1.coef), : len(f2.coef)] @ f2.coef

    error = np.abs(i_num - i_ex)
    norm = np.abs(i_ex)

    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("fun1, fun2, xmin, xmax", prod_list_rc)
def test_basis_projection_rc(fun1, fun2, xmin, xmax):
    prod = fun1 * fun2

    ff1 = fun_from_expr(x, fun1)
    ff2 = fun_from_expr(x, fun2)
    fp = fun_from_expr(x, prod)

    f1 = RealFunction(ff1, xmin, xmax)
    f2 = ComplexFunction(ff2, xmin, xmax)
    p_ex = ComplexFunction(fp, xmin, xmax)

    i_ex = p_ex.integral()

    num_coef = max(len(f1.coef), len(f2.coef))
    basis = Basis1D(num_coef - 1, xmin, xmax)
    P = basis.projection_matrix()
    i_num = f1.coef.T @ P[: len(f1.coef), : len(f2.coef)] @ f2.coef

    error = np.abs(i_num - i_ex)
    norm = np.abs(i_ex)

    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("fun1, fun2, xmin, xmax", prod_list_cc)
def test_basis_projection_cc(fun1, fun2, xmin, xmax):
    prod = fun1 * fun2

    ff1 = fun_from_expr(x, fun1)
    ff2 = fun_from_expr(x, fun2)
    fp = fun_from_expr(x, prod)

    f1 = ComplexFunction(ff1, xmin, xmax)
    f2 = ComplexFunction(ff2, xmin, xmax)
    p_ex = ComplexFunction(fp, xmin, xmax)

    i_ex = p_ex.integral()

    num_coef = max(len(f1.coef), len(f2.coef))
    basis = Basis1D(num_coef - 1, xmin, xmax)
    P = basis.projection_matrix()
    i_num = f1.coef.T @ P[: len(f1.coef), : len(f2.coef)] @ f2.coef

    error = np.abs(i_num - i_ex)
    norm = np.abs(i_ex)

    assert error <= abs_tol + rel_tol * norm
