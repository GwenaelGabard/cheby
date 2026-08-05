import numpy as np
import pytest
from cases import bound_list
from cheby import ComplexConstant, RealConstant

rel_tol = 1.0e-13
abs_tol = 1.0e-13

real_const_list = [0.0, 1.0, -1.0, 2.34, -5.7]
complex_const_list = [0.0j, 1.0j, -2.34j, 1.2 - 3.4j, -0.8 + 2.1j]


@pytest.mark.parametrize("xmin, xmax", bound_list)
@pytest.mark.parametrize("c", real_const_list)
def test_real_constant_value(c, xmin, xmax):
    f = RealConstant(xmin, xmax, c)
    xn = np.linspace(xmin, xmax, 256)
    f_ex = np.full_like(xn, c)
    f_num = f(xn)

    error = np.max(np.abs(f_ex - f_num))
    norm = max(np.abs(c), 1.0)

    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("xmin, xmax", bound_list)
@pytest.mark.parametrize("c", real_const_list)
def test_real_constant_coef(c, xmin, xmax):
    f = RealConstant(xmin, xmax, c)

    assert np.array_equal(f.coef, np.array([c]))
    assert f.start == xmin
    assert f.end == xmax


@pytest.mark.parametrize("xmin, xmax", bound_list)
@pytest.mark.parametrize("c", real_const_list + complex_const_list)
def test_complex_constant_value(c, xmin, xmax):
    f = ComplexConstant(xmin, xmax, c)
    xn = np.linspace(xmin, xmax, 256)
    f_ex = np.full_like(xn, c, dtype=complex)
    f_num = f(xn)

    error = np.max(np.abs(f_ex - f_num))
    norm = max(np.abs(c), 1.0)

    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("xmin, xmax", bound_list)
@pytest.mark.parametrize("c", real_const_list + complex_const_list)
def test_complex_constant_coef(c, xmin, xmax):
    f = ComplexConstant(xmin, xmax, c)

    assert np.array_equal(f.coef, np.array([c], dtype=complex))
    assert f.start == xmin
    assert f.end == xmax
