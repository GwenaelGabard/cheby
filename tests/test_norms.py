import numpy as np
import sympy as sp
import pytest
from cases import complex_case_list, real_case_list, x, fun_from_expr
from cheby import RealFunction, ComplexFunction

rel_tol = 1.0e-12
abs_tol = 1.0e-12
alphas = [0.0, 0.23, 1.0, 10.3]


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list)
def test_normL2_real(fun, xmin, xmax):
    norm_ex = np.sqrt(float(sp.integrate(fun * fun, (x, xmin, xmax)).evalf()))

    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)
    norm_num = f.norm_L2()

    error = np.max(np.abs(norm_ex - norm_num))
    norm = np.max(np.abs(norm_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list + complex_case_list)
def test_normL2_complex(fun, xmin, xmax):
    norm_ex = np.sqrt(
        np.real(complex(sp.integrate(fun * sp.conjugate(fun), (x, xmin, xmax)).evalf()))
    )

    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)
    norm_num = f.norm_L2()

    error = np.max(np.abs(norm_ex - norm_num))
    norm = np.max(np.abs(norm_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list)
def test_normH1_real(fun, xmin, xmax):
    a = float(sp.integrate(fun * fun, (x, xmin, xmax)).evalf())
    dfun = sp.diff(fun, x)
    b = float(sp.integrate(dfun * dfun, (x, xmin, xmax)).evalf())
    norm_ex = np.sqrt(a + b)

    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)
    norm_num = f.norm_H1()

    error = np.max(np.abs(norm_ex - norm_num))
    norm = np.max(np.abs(norm_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list + complex_case_list)
def test_normH1_complex(fun, xmin, xmax):
    a = float(sp.integrate(fun * sp.conjugate(fun), (x, xmin, xmax)).evalf())
    dfun = sp.diff(fun, x)
    b = float(sp.integrate(dfun * sp.conjugate(dfun), (x, xmin, xmax)).evalf())
    norm_ex = np.sqrt(a + b)

    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)
    norm_num = f.norm_H1()

    error = np.max(np.abs(norm_ex - norm_num))
    norm = np.max(np.abs(norm_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list)
@pytest.mark.parametrize("alpha", alphas)
def test_normH1_real2(fun, xmin, xmax, alpha):
    a = float(sp.integrate(fun * fun, (x, xmin, xmax)).evalf())
    dfun = sp.diff(fun, x)
    b = float(sp.integrate(dfun * dfun, (x, xmin, xmax)).evalf())
    norm_ex = np.sqrt(a + alpha * b)

    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)
    norm_num = f.norm_H1(alpha)

    error = np.max(np.abs(norm_ex - norm_num))
    norm = np.max(np.abs(norm_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list + complex_case_list)
@pytest.mark.parametrize("alpha", alphas)
def test_normH1_complex2(fun, xmin, xmax, alpha):
    a = float(sp.integrate(fun * sp.conjugate(fun), (x, xmin, xmax)).evalf())
    dfun = sp.diff(fun, x)
    b = float(sp.integrate(dfun * sp.conjugate(dfun), (x, xmin, xmax)).evalf())
    norm_ex = np.sqrt(a + alpha * b)

    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)
    norm_num = f.norm_H1(alpha)

    error = np.max(np.abs(norm_ex - norm_num))
    norm = np.max(np.abs(norm_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol
