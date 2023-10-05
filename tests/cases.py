import sympy as sp
import numpy as np


def sub_vec(a, b):
    if len(a) < len(b):
        c = -b.copy()
        c[: len(a)] += a
    else:
        c = a.copy()
        c[: len(b)] -= b
    return c


# def fun_from_expr(x, expr):
#     fl = sp.lambdify(x, expr, 'numpy')
#     def fun(xn):
#         v = fl(xn)
#         if isinstance(v, int) or isinstance(v, float) or isinstance(v, complex):
#             return(xn*0.0+v)
#         else:
#             return(v)
#     return(fun)


def fun_from_expr(x, expr):
    return np.vectorize(sp.lambdify(x, expr, "numpy"))


x = sp.Symbol("x")

rfun_list = []
rfun_list.append(0 * x)
rfun_list.append(0 * x + 1.0)
rfun_list.append(0 * x + 2.34)
rfun_list.append(x)
rfun_list.append(1.2 * x)
rfun_list.append(x * x)
rfun_list.append(4.7 * x * x)
rfun_list.append(x**3)
rfun_list.append(5.1 * x**3)
rfun_list.append(x**4)
rfun_list.append(0.9 * x**4)
rfun_list.append(sp.cos(x))
rfun_list.append(sp.cos(4.1 * x))
rfun_list.append(sp.cos(10 * x))
rfun_list.append(sp.cos(30 * x))
rfun_list.append(sp.sin(x))
rfun_list.append(sp.sin(4.1 * x))
rfun_list.append(sp.sin(10 * x))
rfun_list.append(sp.sin(30 * x))
rfun_list.append(sp.exp(x))
rfun_list.append(sp.exp(0.4 * x))
rfun_list.append(sp.exp(3.2 * x))
rfun_list.append(sp.exp(-x))
rfun_list.append(sp.exp(-0.4 * x))
rfun_list.append(sp.exp(-3.2 * x))
rfun_list.append(sp.exp(-(x**2)))
rfun_list.append(sp.exp(-0.4 * x**2))
rfun_list.append(sp.exp(-2.2 * x**2))

cfun_list = []
cfun_list.append(0 * x + 1.0j)
cfun_list.append(0 * x + 2.34j)
cfun_list.append(1.2j * x)
cfun_list.append(4.7j * x * x)
cfun_list.append(5.1j * x**3)
cfun_list.append(0.9j * x**4)
cfun_list.append(sp.exp(1j * x))
cfun_list.append(sp.exp(4.1j * x))
cfun_list.append(sp.exp(10j * x))
cfun_list.append(sp.exp(30j * x))

bound_list = []
bound_list.append((-1.0, 1.0))
bound_list.append((0.0, 1.0))
bound_list.append((-1.0, 0.0))
bound_list.append((-3, -0.2))
bound_list.append((-0.6, -2.1))
bound_list.append((-2.4, 1.8))
bound_list.append((0.3, 4.1))

real_case_list = []
for f in rfun_list:
    for b in bound_list:
        real_case_list.append((f, b[0], b[1]))

complex_case_list = []
for f in cfun_list:
    for b in bound_list:
        complex_case_list.append((f, b[0], b[1]))

prod_list_rr = []
for f1 in rfun_list:
    for f2 in rfun_list:
        for b in bound_list:
            prod_list_rr.append((f1, f2, b[0], b[1]))

prod_list_cr = []
for f1 in cfun_list:
    for f2 in rfun_list:
        for b in bound_list:
            prod_list_cr.append((f1, f2, b[0], b[1]))

prod_list_rc = []
for f1 in rfun_list:
    for f2 in cfun_list:
        for b in bound_list:
            prod_list_rc.append((f1, f2, b[0], b[1]))

prod_list_cc = []
for f1 in cfun_list:
    for f2 in cfun_list:
        for b in bound_list:
            prod_list_cc.append((f1, f2, b[0], b[1]))
