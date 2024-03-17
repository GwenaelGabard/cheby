from itertools import product
import numpy as np
import pytest
from cases import bound_list
from cheby import Basis1D

rel_tol = 1.0e-12
abs_tol = 1.0e-12

order_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 25, 34, 123]
degree_list = [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "order, bounds, degree", product(order_list, bound_list, degree_list)
)
def test_basis_derivative(order, bounds, degree):
    basis = Basis1D(order, bounds[0], bounds[1])
    xi = np.linspace(-1, 1, 56)
    x = (xi + 1) / 2.0 * (bounds[1] - bounds[0]) + bounds[0]
    V_ex = basis.eval(x)
    dV = basis.derivatives(x, degree)
    D = basis.diff_matrix()

    for d in range(degree + 1):
        V = dV[d]
        error = np.max(np.abs(V - V_ex))
        norm = np.max(np.abs(V_ex))

        if norm == 0.0:
            assert error < abs_tol
        else:
            assert error / norm < rel_tol

        V_ex = V_ex @ D
