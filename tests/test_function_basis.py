import pytest
from cases import complex_case_list, fun_from_expr, real_case_list, x
from cheby import ComplexFunction, RealFunction


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list)
def test_basis_real(fun, xmin, xmax):
    ff = fun_from_expr(x, fun)
    f = RealFunction(ff, xmin, xmax)
    b = f.basis()

    assert b.start == xmin
    assert b.end == xmax
    assert b.order == len(f.coef) - 1


@pytest.mark.parametrize("fun, xmin, xmax", real_case_list + complex_case_list)
def test_basis_complex(fun, xmin, xmax):
    ff = fun_from_expr(x, fun)
    f = ComplexFunction(ff, xmin, xmax)
    b = f.basis()

    assert b.start == xmin
    assert b.end == xmax
    assert b.order == len(f.coef) - 1
