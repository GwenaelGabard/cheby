import numpy as np
import pytest
from cheby import RealFunction

rel_tol = 1.0e-6
abs_tol = 1.0e-6

order = 30
inverse_order = 25

# Each case is a strictly monotonic (increasing or decreasing) function with a
# known closed-form inverse.
# Intervals are chosen away from any point where the inverse's derivative blows up.
cases = [
    pytest.param(lambda x: x, -1.0, 1.0, lambda y: y, id="identity"),
    pytest.param(lambda x: x, -2.4, 1.8, lambda y: y, id="identity-shifted"),
    pytest.param(lambda x: -x, -1.0, 1.0, lambda y: -y, id="negation"),
    pytest.param(lambda x: -x, -2.4, 1.8, lambda y: -y, id="negation-shifted"),
    pytest.param(
        lambda x: 2 * x + 3, -1.0, 1.0, lambda y: (y - 3) / 2, id="affine-increasing"
    ),
    pytest.param(
        lambda x: -3 * x + 1, -2.0, 2.0, lambda y: (1 - y) / 3, id="affine-decreasing"
    ),
    pytest.param(
        lambda x: np.exp(x), -1.0, 1.0, lambda y: np.log(y), id="exp-increasing"
    ),
    pytest.param(
        lambda x: -np.exp(x), -1.0, 1.0, lambda y: np.log(-y), id="neg-exp-decreasing"
    ),
    pytest.param(
        lambda x: np.sin(x), -1.0, 1.0, lambda y: np.arcsin(y), id="sin-increasing"
    ),
    pytest.param(
        lambda x: np.cos(x),
        1.0,
        np.pi - 1.0,
        lambda y: np.arccos(y),
        id="cos-decreasing",
    ),
    pytest.param(lambda x: x**3, 1.0, 2.0, lambda y: np.cbrt(y), id="cube-increasing"),
    pytest.param(
        lambda x: 1.0 / x, 1.0, 3.0, lambda y: 1.0 / y, id="reciprocal-decreasing"
    ),
]


@pytest.mark.parametrize("ff, xmin, xmax, ginv", cases)
def test_inverse(ff, xmin, xmax, ginv):
    f = RealFunction(ff, xmin, xmax, order)
    g = f.inverse(inverse_order)

    fa, fb = ff(xmin), ff(xmax)
    assert g.start == pytest.approx(fa)
    assert g.end == pytest.approx(fb)

    ys = np.linspace(min(fa, fb), max(fa, fb), 30)
    error = np.max(np.abs(g(ys) - ginv(ys)))
    norm = np.max(np.abs(ginv(ys)))
    assert error <= abs_tol + rel_tol * norm


@pytest.mark.parametrize("N", [1, 2, 3, 4, 5])
def test_inverse_small_n_accuracy(N):
    def ff(x):
        return 2 * x + 3

    def ginv(y):
        return (y - 3) / 2

    f = RealFunction(ff, -1.0, 1.0, order)
    g = f.inverse(N)

    assert len(g.coef) == N + 1

    ys = np.linspace(ff(-1.0), ff(1.0), 10)
    error = np.max(np.abs(g(ys) - ginv(ys)))
    assert error <= abs_tol


def test_inverse_n_zero_does_not_crash():
    f = RealFunction(lambda x: 2 * x + 3, -1.0, 1.0, order)
    g = f.inverse(0)

    assert len(g.coef) == 1
