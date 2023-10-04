from itertools import product
import numpy as np
import pytest
from cheby import Basis1D

rtol = 1e-10
atol = 1e-12

order_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 25, 34, 123]
num_list = [2, 3, 4, 5, 6, 7, 8]
range_list = [(-1.0, 1.0), (0.0, 1.0), (-1.0, 0.0), (-3, -0.2), (-0.6, 2.1)]


@pytest.mark.parametrize("order, range", product(order_list, range_list))
def test_nolength(order, range):
    basis = Basis1D(order, range[0], range[1])
    p = basis.points2()
    assert len(p) == order + 1


@pytest.mark.parametrize("order, range", product(order_list, range_list))
def test_length0(order, range):
    basis = Basis1D(order, range[0], range[1])
    p = basis.points2(0)
    assert len(p) == 0


@pytest.mark.parametrize("order, range", product(order_list, range_list))
def test_length1(order, range):
    basis = Basis1D(order, range[0], range[1])
    p = basis.points2(1)
    assert len(p) == 0


@pytest.mark.parametrize(
    "order, range, num_points", product(order_list, range_list, num_list)
)
def test_length(order, range, num_points):
    basis = Basis1D(order, range[0], range[1])
    p = basis.points2(num_points)
    assert len(p) == num_points


@pytest.mark.parametrize(
    "order, range, num_points", product(order_list, range_list, num_list)
)
def test_value(order, range, num_points):
    basis = Basis1D(order, range[0], range[1])
    p = basis.points2(num_points)
    xi = -np.cos(np.arange(num_points) / (num_points - 1) * np.pi)
    p_ref = (xi + 1) / 2.0 * (range[1] - range[0]) + range[0]
    assert np.isclose(p, p_ref, rtol, atol).all()
