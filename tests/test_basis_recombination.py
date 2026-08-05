import numpy as np
import pytest
from cases import bound_list
from cheby import Basis1D

rel_tol = 1.0e-12
abs_tol = 1.0e-12

order_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 25, 34, 123]


@pytest.mark.parametrize("bounds", bound_list)
@pytest.mark.parametrize("order", order_list)
def test_basis_dirichlet(order, bounds):
    xmin, xmax = bounds
    basis = Basis1D(order, xmin, xmax)

    x = np.array([xmin, xmax])
    T = basis.eval(x)
    Phi = T @ basis.dirichlet()

    expected = np.zeros((2, order + 1))
    expected[0, 0] = 1.0
    expected[1, 1] = 1.0

    error = np.max(np.abs(Phi - expected))
    assert error <= abs_tol + rel_tol


@pytest.mark.parametrize("bounds", bound_list)
@pytest.mark.parametrize("order", order_list)
def test_basis_neumann(order, bounds):
    xmin, xmax = bounds
    basis = Basis1D(order, xmin, xmax)

    x = np.array([xmin, xmax])
    dT = basis.derivatives(x, 1)[1]
    dPhi = dT @ basis.neumann()

    expected = np.zeros((2, order + 1))
    expected[0, 0] = 2.0 / (xmax - xmin)
    expected[1, 1] = 2.0 / (xmax - xmin)

    error = np.max(np.abs(dPhi - expected))
    assert error <= abs_tol + rel_tol
