from itertools import product
import numpy as np
import pytest
from cheby import Basis1D
from numpy.polynomial.chebyshev import chebvander

rel_tol = 1.0e-12
abs_tol = 1.0e-12

order_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 25, 34, 123]
range_list = [(-1.0, 1.0), (0.0, 1.0), (-1.0, 0.0), (-3, -0.2), (-0.6, 2.1)]


@pytest.mark.parametrize("order, bounds", product(order_list, range_list))
def test_basis_eval(order, bounds):
    basis = Basis1D(order, bounds[0], bounds[1])
    xi = np.linspace(-1, 1, 56)
    x = (xi + 1) / 2.0 * (bounds[1] - bounds[0]) + bounds[0]
    V_ex = chebvander(xi, order)
    V = basis.eval(x)

    error = np.max(np.abs(V - V_ex))
    norm = np.max(np.abs(V_ex))

    if norm == 0.0:
        assert error < abs_tol
    else:
        assert error / norm < rel_tol
