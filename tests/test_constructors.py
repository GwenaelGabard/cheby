import numpy as np
import pytest
from cases import bound_list
from cheby import Basis1D, ComplexFunction, RealFunction
from numpy.polynomial.chebyshev import chebval

rel_tol = 1.0e-10
abs_tol = 1.0e-10

order_list = [0, 1, 2, 3, 5, 10, 20]

real_coef_list = [
    np.array([1.0]),
    np.array([0.3, -1.2]),
    np.array([0.3, -1.2, 0.7, 2.1, -0.4]),
    np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
]

complex_coef_list = [
    np.array([1.0 + 0.0j]),
    np.array([0.3 + 0.1j, -1.2 - 0.4j]),
    np.array([0.3 + 0.1j, -1.2 - 0.4j, 0.7 + 2.0j, 2.1 - 0.3j, -0.4 + 0.9j]),
]


@pytest.mark.parametrize("bounds", bound_list)
@pytest.mark.parametrize("coef", real_coef_list)
def test_direct_coef_real(coef, bounds):
    xmin, xmax = bounds
    f = RealFunction(xmin, xmax, coef)

    assert np.array_equal(f.coef, coef)
    assert f.start == xmin
    assert f.end == xmax

    xs = np.linspace(xmin, xmax, 20)
    xi = 2 * (xs - xmin) / (xmax - xmin) - 1
    ref = chebval(xi, coef)

    error = np.max(np.abs(f(xs) - ref))
    norm = np.max(np.abs(ref))
    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("bounds", bound_list)
@pytest.mark.parametrize("coef", real_coef_list + complex_coef_list)
def test_direct_coef_complex(coef, bounds):
    xmin, xmax = bounds
    f = ComplexFunction(xmin, xmax, coef.astype(complex))

    assert np.array_equal(f.coef, coef.astype(complex))
    assert f.start == xmin
    assert f.end == xmax

    xs = np.linspace(xmin, xmax, 20)
    xi = 2 * (xs - xmin) / (xmax - xmin) - 1
    ref = chebval(xi, coef.astype(complex))

    error = np.max(np.abs(f(xs) - ref))
    norm = np.max(np.abs(ref))
    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("bounds", bound_list)
@pytest.mark.parametrize("order", order_list)
def test_explicit_order_coef_count_real(order, bounds):
    xmin, xmax = bounds
    f = RealFunction(lambda xn: np.cos(3 * xn), xmin, xmax, order)
    assert len(f.coef) == order + 1


@pytest.mark.parametrize("bounds", bound_list)
@pytest.mark.parametrize("order", order_list)
def test_explicit_order_coef_count_complex(order, bounds):
    xmin, xmax = bounds
    f = ComplexFunction(lambda xn: np.exp(1j * xn), xmin, xmax, order)
    assert len(f.coef) == order + 1


@pytest.mark.parametrize("bounds", bound_list)
def test_explicit_order_accuracy_real(bounds):
    xmin, xmax = bounds

    def ff(xn):
        return np.cos(3 * xn)

    f = RealFunction(ff, xmin, xmax, 25)
    xs = np.linspace(xmin, xmax, 50)

    error = np.max(np.abs(f(xs) - ff(xs)))
    norm = np.max(np.abs(ff(xs)))
    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("bounds", bound_list)
def test_explicit_order_accuracy_complex(bounds):
    xmin, xmax = bounds

    def ff(xn):
        return np.exp(1j * xn)

    f = ComplexFunction(ff, xmin, xmax, 25)
    xs = np.linspace(xmin, xmax, 50)

    error = np.max(np.abs(f(xs) - ff(xs)))
    norm = np.max(np.abs(ff(xs)))
    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("bounds", bound_list)
def test_explicit_order_exact_polynomial_real(bounds):
    xmin, xmax = bounds

    def ff(xn):
        return xn**3 - 2 * xn + 1

    f = RealFunction(ff, xmin, xmax, 3)
    xs = np.linspace(xmin, xmax, 20)

    error = np.max(np.abs(f(xs) - ff(xs)))
    norm = np.max(np.abs(ff(xs)))
    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("bounds", bound_list)
def test_explicit_order_exact_polynomial_complex(bounds):
    xmin, xmax = bounds

    def ff(xn):
        return (1 + 2j) * xn**3 - 2 * xn + 1j

    f = ComplexFunction(ff, xmin, xmax, 3)
    xs = np.linspace(xmin, xmax, 20)

    error = np.max(np.abs(f(xs) - ff(xs)))
    norm = np.max(np.abs(ff(xs)))
    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("bounds", bound_list)
def test_explicit_order_zero_is_midpoint_real(bounds):
    xmin, xmax = bounds

    def ff(xn):
        return xn**2 + 3.0

    f = RealFunction(ff, xmin, xmax, 0)
    midpoint = (xmin + xmax) / 2.0

    assert len(f.coef) == 1
    assert f.coef[0] == pytest.approx(ff(np.array([midpoint]))[0])


@pytest.mark.parametrize("bounds", bound_list)
def test_explicit_order_zero_is_midpoint_complex(bounds):
    xmin, xmax = bounds

    def ff(xn):
        return xn**2 + 3.0j

    f = ComplexFunction(ff, xmin, xmax, 0)
    midpoint = (xmin + xmax) / 2.0

    assert len(f.coef) == 1
    assert f.coef[0] == pytest.approx(ff(np.array([midpoint]))[0])


@pytest.mark.parametrize("bounds", bound_list)
def test_auto_order_matches_explicit_order_real(bounds):
    xmin, xmax = bounds

    def ff(xn):
        return np.cos(3 * xn)

    f = RealFunction(ff, xmin, xmax)
    order = len(f.coef) - 1

    g = RealFunction(ff, xmin, xmax, order)

    assert np.max(np.abs(f.coef - g.coef)) <= abs_tol


@pytest.mark.parametrize("bounds", bound_list)
def test_auto_order_matches_explicit_order_complex(bounds):
    xmin, xmax = bounds

    def ff(xn):
        return np.exp(1j * xn)

    f = ComplexFunction(ff, xmin, xmax)
    order = len(f.coef) - 1

    g = ComplexFunction(ff, xmin, xmax, order)

    assert np.max(np.abs(f.coef - g.coef)) <= abs_tol


@pytest.mark.parametrize("bounds", bound_list)
@pytest.mark.parametrize("order", order_list)
def test_basis1d_constructor(order, bounds):
    xmin, xmax = bounds
    b = Basis1D(order, xmin, xmax)

    assert b.order == order
    assert b.start == xmin
    assert b.end == xmax


@pytest.mark.parametrize("order", order_list)
def test_basis1d_constructor_default_bounds(order):
    b = Basis1D(order)

    assert b.order == order
    assert b.start == -1.0
    assert b.end == 1.0
