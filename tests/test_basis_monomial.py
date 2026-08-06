import numpy as np
import pytest
from cases import bound_list
from cheby import Basis1D

rel_tol = 1.0e-9
abs_tol = 1.0e-9

# These orders are small because the Chebyshev-to-monomial conversion
# tends to be ill-conditioned for high orders.
order_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]


@pytest.mark.parametrize("bounds", bound_list)
@pytest.mark.parametrize("order", order_list)
def test_basis_monomial(order, bounds):
    xmin, xmax = bounds
    basis = Basis1D(order, xmin, xmax)

    x = np.linspace(xmin, xmax, 30)
    xi = 2 * (x - xmin) / (xmax - xmin) - 1
    T = basis.eval(x)

    monomials = basis.monomial_matrix()
    vandermonde = xi[:, None] ** np.arange(order + 1)[None, :]
    reconstructed = vandermonde @ monomials

    error = np.max(np.abs(reconstructed - T))
    assert error <= abs_tol + rel_tol
