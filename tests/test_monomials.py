import numpy as np
import pytest
from cases import bound_list
from cheby import Basis1D, ComplexFunction, RealFunction

rel_tol = 1.0e-9
abs_tol = 1.0e-9

# These coefficient vectors are kept short because the Chebyshev-to-monomial conversion
# tends to be ill-conditioned for high orders.
real_coef_list = [
    np.array([]),
    np.array([3.0]),
    np.array([0.3, -1.2]),
    np.array([0.3, -1.2, 0.7, 2.1, -0.4]),
    np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
]

complex_coef_list = [
    np.array([], dtype=complex),
    np.array([3.0 + 1.0j]),
    np.array([0.3 + 0.1j, -1.2 - 0.4j]),
    np.array([0.3 + 0.1j, -1.2 - 0.4j, 0.7 + 2.0j, 2.1 - 0.3j, -0.4 + 0.9j]),
]


def reference_monomials(coef, xmin, xmax):
    if len(coef) == 0:
        return np.array([], dtype=coef.dtype)
    basis = Basis1D(len(coef) - 1, xmin, xmax)
    return basis.monomial_matrix() @ coef


@pytest.mark.parametrize("bounds", bound_list)
@pytest.mark.parametrize("coef", real_coef_list)
def test_monomials_real(coef, bounds):
    xmin, xmax = bounds
    f = RealFunction(xmin, xmax, coef)

    monomials = f.monomials()
    ref = reference_monomials(coef, xmin, xmax)

    assert len(monomials) == len(coef)
    if len(ref) > 0:
        error = np.max(np.abs(monomials - ref))
        norm = np.max(np.abs(ref))
        assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("bounds", bound_list)
@pytest.mark.parametrize("coef", real_coef_list + complex_coef_list)
def test_monomials_complex(coef, bounds):
    xmin, xmax = bounds
    coef = coef.astype(complex)
    f = ComplexFunction(xmin, xmax, coef)

    monomials = f.monomials()
    ref = reference_monomials(coef, xmin, xmax)

    assert len(monomials) == len(coef)
    if len(ref) > 0:
        error = np.max(np.abs(monomials - ref))
        norm = np.max(np.abs(ref))
        assert error <= abs_tol + rel_tol * norm
