#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cheby/cheby.hpp"
#include "cheby_version.hpp"

namespace nb = nanobind;
using namespace cheby;

void init_cheby(nb::module_& m) {
    nb::class_<Basis1D>(m, "Basis1D",
                        "A basis of Chebyshev polynomials of the first kind.\n"
                        "\n"
                        "A basis is defined by its order (highest polynomial degree) and the\n"
                        "interval ``[start, end]`` it is defined over. It represents the\n"
                        "basis functions themselves, independent of any function's\n"
                        "coefficients.")
        .def(nb::init<int, double, double>(), nb::arg("order"), nb::arg("start") = -1.0,
             nb::arg("end") = 1.0,
             "Construct a Chebyshev polynomial basis.\n"
             "\n"
             "Args:\n"
             "    order: The order (highest degree) of the basis.\n"
             "    start: The start of the interval.\n"
             "    end: The end of the interval.")
        .def_ro("order", &Basis1D::order, "The order of the basis.")
        .def_ro("start", &Basis1D::xmin, "The start of the interval.")
        .def_ro("end", &Basis1D::xmax, "The end of the interval.")
        .def("points1", &Basis1D::Points1, nb::arg("num_points") = -1,
             "Chebyshev points of the first kind.\n"
             "\n"
             "Args:\n"
             "    num_points: The number of points to generate. Defaults to\n"
             "        ``order + 1``.\n"
             "\n"
             "Returns:\n"
             "    A vector with the coordinates of the points.")
        .def("points2", &Basis1D::Points2, nb::arg("num_points") = -1,
             "Chebyshev points of the second kind.\n"
             "\n"
             "Args:\n"
             "    num_points: The number of points to generate. Defaults to\n"
             "        ``order + 1``.\n"
             "\n"
             "Returns:\n"
             "    A vector with the coordinates of the points.")
        .def("eval", &Basis1D::Eval, nb::arg("x"),
             "Evaluate the Chebyshev polynomials of the first kind.\n"
             "\n"
             "Args:\n"
             "    x: The points at which to evaluate the polynomials.\n"
             "\n"
             "Returns:\n"
             "    The matrix of polynomial values at the given points (each\n"
             "    row is a point, each column is a polynomial).")
        .def("derivatives", &Basis1D::Derivatives, nb::arg("x"), nb::arg("D"),
             "Evaluate derivatives of the Chebyshev polynomials of the first\n"
             "kind.\n"
             "\n"
             "Args:\n"
             "    x: The points at which to evaluate the polynomials.\n"
             "    D: The maximum order of the derivatives to evaluate.\n"
             "\n"
             "Returns:\n"
             "    A list of matrices of polynomial derivatives, one per\n"
             "    derivative order up to ``D`` (each row is a point, each\n"
             "    column is a polynomial).")
        .def("dirichlet", &Basis1D::DirichletMatrix,
             "The matrix for the Dirichlet basis recombination.\n"
             "\n"
             "The recombined basis is zero at the end points, except for the\n"
             "first two basis functions. The first function is 1 and 0 at\n"
             "the start and end points, respectively. The second function is\n"
             "0 and 1 at the start and end points, respectively.\n"
             "\n"
             "Returns:\n"
             "    The square recombination matrix.")
        .def("neumann", &Basis1D::NeumannMatrix,
             "The matrix for the Neumann basis recombination.\n"
             "\n"
             "The recombined basis has zero derivatives at the end points,\n"
             "except for the first and second basis functions. The first\n"
             "function has derivative 1 and 0 at the start and end points,\n"
             "respectively. The second function has derivative 0 and 1 at the\n"
             "start and end points, respectively.\n"
             "\n"
             "Returns:\n"
             "    The square recombination matrix.")
        .def("diff_matrix", &Basis1D::DiffMatrix,
             "The differentiation matrix for the Chebyshev polynomials of\n"
             "the first kind.\n"
             "\n"
             "Returns:\n"
             "    The square differentiation matrix.")
        .def("projection_matrix", &Basis1D::ProjectionMatrix,
             "The projection matrix, or Gram matrix, for the Chebyshev polynomials of the\n"
             "first kind.\n"
             "\n"
             "Returns:\n"
             "    The square projection matrix.")
        .def("monomial_matrix", &Basis1D::MonomialMatrix,
             "The matrix converting Chebyshev coefficients into polynomial\n"
             "coefficients.\n"
             "\n"
             "Returns:\n"
             "    A square matrix. The nth column contains the polynomial\n"
             "    coefficients of the nth Chebyshev polynomial.");

    nb::class_<RealFunction>(m, "RealFunction",
                             "A real-valued function represented as a Chebyshev series.\n"
                             "\n"
                             "The function is represented by a vector of Chebyshev coefficients\n"
                             "(``coef``) over the interval ``[start, end]``.")
        .def(nb::init<std::function<RealFunction::ValueVector(RealFunction::ParamVector)>, double,
                      double, int>(),
             nb::arg("f"), nb::arg("start"), nb::arg("end"), nb::arg("order") = -1,
             "Construct a Chebyshev representation of a function by sampling\n"
             "it at Chebyshev points.\n"
             "\n"
             "Args:\n"
             "    f: The function to represent, evaluated on a vector of\n"
             "        points and returning a vector of values.\n"
             "    start: The start of the interval.\n"
             "    end: The end of the interval.\n"
             "    order: The order of the Chebyshev series. If negative,\n"
             "        the order is chosen automatically so\n"
             "        that the coefficient tail decays below the relative\n"
             "        tolerance.")
        .def(nb::init<double, double, RealFunction::CoefVector>(), nb::arg("start"), nb::arg("end"),
             nb::arg("coef"),
             "Construct a Chebyshev representation of a function directly\n"
             "from its coefficients.\n"
             "\n"
             "Args:\n"
             "    start: The start of the interval.\n"
             "    end: The end of the interval.\n"
             "    coef: The vector of Chebyshev coefficients.")
        .def("__call__", &RealFunction::Eval, nb::arg("x"),
             "Evaluate the function at a number of points.\n"
             "\n"
             "Args:\n"
             "    x: The points at which to evaluate the function.\n"
             "\n"
             "Returns:\n"
             "    The values of the function at the given points.")
        .def("__add__", &Add<RealFunction, RealFunction, RealFunction>, "Add two functions.")
        .def("__add__", &Add<RealFunction, ComplexFunction, ComplexFunction>,
             "Add a real and a complex function.")
        .def("__sub__", &Sub<RealFunction, RealFunction, RealFunction>, "Subtract two functions.")
        .def("__sub__", &Sub<RealFunction, ComplexFunction, ComplexFunction>,
             "Subtract a complex function from a real function.")
        .def("__mul__", &Multiply<RealFunction, RealFunction, RealFunction>, "Multiply two functions.")
        .def("__mul__", &Multiply<RealFunction, ComplexFunction, ComplexFunction>,
             "Multiply a real and a complex function.")
        .def_ro("start", &RealFunction::xmin, "The start of the interval.")
        .def_ro("end", &RealFunction::xmax, "The end of the interval.")
        .def_ro("coef", &RealFunction::coef, "The vector of Chebyshev coefficients.")
        .def("basis", &RealFunction::GetBasis,
             "The Chebyshev basis of the function.\n"
             "\n"
             "Returns:\n"
             "    A :class:`Basis1D` with the same order and interval as the\n"
             "    function.")
        .def("tail_length", &RealFunction::TailLength,
             "The number of trailing coefficients that are negligible.\n"
             "\n"
             "Returns:\n"
             "    The number of coefficients, counted from the end of the\n"
             "    coefficient vector, that fall below the relative\n"
             "    tolerance.")
        .def("trim", &RealFunction::Trim,
             "Trim the negligible trailing coefficients from the function,\n"
             "in place.")
        .def("derivative", &RealFunction::Derivative,
             "The derivative of the function.\n"
             "\n"
             "Returns:\n"
             "    A new function representing the derivative.")
        .def("primitive", &RealFunction::Primitive,
             "The primitive (anti-derivative) of the function.\n"
             "\n"
             "Returns:\n"
             "    A new function representing the primitive.")
        .def("integral", nb::overload_cast<>(&RealFunction::Integral, nb::const_),
             "The integral of the function over the whole interval.\n"
             "\n"
             "Returns:\n"
             "    The integral of the function over ``[start, end]``.")
        .def("integral",
             nb::overload_cast<const RealFunction::Parameter, const RealFunction::Parameter>(
                 &RealFunction::Integral, nb::const_),
             "The integral of the function over a subinterval.\n"
             "\n"
             "Args:\n"
             "    a: The start of the subinterval.\n"
             "    b: The end of the subinterval.\n"
             "\n"
             "Returns:\n"
             "    The integral of the function over ``[a, b]``.")
        .def("real", &RealFunction::Real,
             "The real part of the function (a copy of the function itself).")
        .def("imag", &RealFunction::Imag, "The imaginary part of the function (identically zero).")
        .def("conj", &RealFunction::Conjugate,
             "The complex conjugate of the function (a copy of the function\n"
             "itself).")
        .def("norm_L2", &RealFunction::NormL2,
             "The L2 norm of the function.\n"
             "\n"
             "Returns:\n"
             "    The L2 norm of the function over its interval.")
        .def("norm_H1", &RealFunction::NormH1, nb::arg("alpha") = 1.0,
             "The H1 norm of the function.\n"
             "\n"
             "Args:\n"
             "    alpha: The weight of the derivative term in the norm.\n"
             "\n"
             "Returns:\n"
             "    The H1 norm of the function over its interval.")
        .def("colleague", &RealFunction::ColleagueMatrix,
             "The colleague matrix of the function.\n"
             "\n"
             "Returns:\n"
             "    The colleague matrix, whose eigenvalues give the roots of\n"
             "    the function.")
        .def("roots", &RealFunction::Roots,
             "The roots of the function.\n"
             "\n"
             "Returns:\n"
             "    A vector of the real roots of the function within its\n"
             "    interval.")
        .def("extrema", &RealFunction::Extrema,
             "The extrema of the function.\n"
             "\n"
             "Returns:\n"
             "    A vector of the points within the interval where the\n"
             "    derivative of the function vanishes.")
        .def("pow", &RealFunction::Power, nb::arg("n"),
             "Raise the function to an integer power.\n"
             "\n"
             "Args:\n"
             "    n: The power to which to raise the function.\n"
             "\n"
             "Returns:\n"
             "    A new function representing ``self`` raised to the power\n"
             "    ``n``.")
        .def("__pow__", &RealFunction::Power, nb::arg("n"),
             "Raise the function to an integer power. See :meth:`pow`.")
        .def("monomials", &RealFunction::Monomials,
             "The coefficients of the corresponding polynomial.\n"
             "\n"
             "Returns:\n"
             "    The coefficients of the polynomial (in the monomial\n"
             "    basis) equivalent to this Chebyshev series.")
        .def("inverse", &RealFunction::Inverse, nb::arg("N"),
             "The functional inverse of the function.\n"
             "\n"
             "Assumes the function is monotonic over its interval.\n"
             "\n"
             "Args:\n"
             "    N: The number of coefficients of the inverse function.\n"
             "\n"
             "Returns:\n"
             "    A new function representing the inverse of ``self``.")
        .def("product_matrix", &RealFunction::ProductMatrix, nb::arg("order"), nb::arg("rows") = -1,
             "The product matrix of the function.\n"
             "\n"
             "The product matrix is a matrix that, when multiplied by the\n"
             "vector of coefficients of another function, gives the\n"
             "coefficients of the product of the two functions.\n"
             "\n"
             "Args:\n"
             "    order: The order of the product matrix.\n"
             "    rows: The number of rows of the product matrix. Defaults\n"
             "        to a size large enough to hold the full product.\n"
             "\n"
             "Returns:\n"
             "    The product matrix of the function.");

    nb::class_<ComplexFunction>(m, "ComplexFunction",
                                "A complex-valued function represented as a Chebyshev series.\n"
                                "\n"
                                "The function is represented by a vector of Chebyshev coefficients\n"
                                "(``coef``) over the interval ``[start, end]``.")
        .def(nb::init<std::function<ComplexFunction::ValueVector(ComplexFunction::ParamVector)>, double,
                      double, int>(),
             nb::arg("f"), nb::arg("start"), nb::arg("end"), nb::arg("order") = -1,
             "Construct a Chebyshev representation of a function by sampling\n"
             "it at Chebyshev points.\n"
             "\n"
             "Args:\n"
             "    f: The function to represent, evaluated on a vector of\n"
             "        points and returning a vector of values.\n"
             "    start: The start of the interval.\n"
             "    end: The end of the interval.\n"
             "    order: The order of the Chebyshev series. If negative,\n"
             "        the order is chosen automatically so\n"
             "        that the coefficient tail decays below the relative\n"
             "        tolerance.")
        .def(nb::init<double, double, ComplexFunction::CoefVector>(), nb::arg("start"), nb::arg("end"),
             nb::arg("coef"),
             "Construct a Chebyshev representation of a function directly\n"
             "from its coefficients.\n"
             "\n"
             "Args:\n"
             "    start: The start of the interval.\n"
             "    end: The end of the interval.\n"
             "    coef: The vector of Chebyshev coefficients.")
        .def("__call__", &ComplexFunction::Eval, nb::arg("x"),
             "Evaluate the function at a number of points.\n"
             "\n"
             "Args:\n"
             "    x: The points at which to evaluate the function.\n"
             "\n"
             "Returns:\n"
             "    The values of the function at the given points.")
        .def("__add__", &Add<ComplexFunction, RealFunction, ComplexFunction>,
             "Add a complex and a real function.")
        .def("__add__", &Add<ComplexFunction, ComplexFunction, ComplexFunction>, "Add two functions.")
        .def("__sub__", &Sub<ComplexFunction, RealFunction, ComplexFunction>,
             "Subtract a real function from a complex function.")
        .def("__sub__", &Sub<ComplexFunction, ComplexFunction, ComplexFunction>,
             "Subtract two functions.")
        .def("__mul__", &Multiply<ComplexFunction, ComplexFunction, ComplexFunction>,
             "Multiply two functions.")
        .def("__mul__", &Multiply<ComplexFunction, RealFunction, ComplexFunction>,
             "Multiply a complex and a real function.")
        .def_ro("start", &ComplexFunction::xmin, "The start of the interval.")
        .def_ro("end", &ComplexFunction::xmax, "The end of the interval.")
        .def_ro("coef", &ComplexFunction::coef, "The vector of Chebyshev coefficients.")
        .def("basis", &ComplexFunction::GetBasis,
             "The Chebyshev basis of the function.\n"
             "\n"
             "Returns:\n"
             "    A :class:`Basis1D` with the same order and interval as the\n"
             "    function.")
        .def("tail_length", &ComplexFunction::TailLength,
             "The number of trailing coefficients that are negligible.\n"
             "\n"
             "Returns:\n"
             "    The number of coefficients, counted from the end of the\n"
             "    coefficient vector, that fall below the relative\n"
             "    tolerance.")
        .def("trim", &ComplexFunction::Trim,
             "Trim the negligible trailing coefficients from the function,\n"
             "in place.")
        .def("derivative", &ComplexFunction::Derivative,
             "The derivative of the function.\n"
             "\n"
             "Returns:\n"
             "    A new function representing the derivative.")
        .def("primitive", &ComplexFunction::Primitive,
             "The primitive (anti-derivative) of the function.\n"
             "\n"
             "Returns:\n"
             "    A new function representing the primitive.")
        .def("integral", nb::overload_cast<>(&ComplexFunction::Integral, nb::const_),
             "The integral of the function over the whole interval.\n"
             "\n"
             "Returns:\n"
             "    The integral of the function over ``[start, end]``.")
        .def("integral",
             nb::overload_cast<const ComplexFunction::Parameter, const ComplexFunction::Parameter>(
                 &ComplexFunction::Integral, nb::const_),
             "The integral of the function over a subinterval.\n"
             "\n"
             "Args:\n"
             "    a: The start of the subinterval.\n"
             "    b: The end of the subinterval.\n"
             "\n"
             "Returns:\n"
             "    The integral of the function over ``[a, b]``.")
        .def("real", &ComplexFunction::Real,
             "The real part of the function.\n"
             "\n"
             "Returns:\n"
             "    A new :class:`RealFunction` with the real part of the\n"
             "    coefficients.")
        .def("imag", &ComplexFunction::Imag,
             "The imaginary part of the function.\n"
             "\n"
             "Returns:\n"
             "    A new :class:`RealFunction` with the imaginary part of the\n"
             "    coefficients.")
        .def("conj", &ComplexFunction::Conjugate,
             "The complex conjugate of the function.\n"
             "\n"
             "Returns:\n"
             "    A new function with the conjugated coefficients.")
        .def("norm_L2", &ComplexFunction::NormL2,
             "The L2 norm of the function.\n"
             "\n"
             "Returns:\n"
             "    The L2 norm of the function over its interval.")
        .def("norm_H1", &ComplexFunction::NormH1, nb::arg("alpha") = 1.0,
             "The H1 norm of the function.\n"
             "\n"
             "Args:\n"
             "    alpha: The weight of the derivative term in the norm.\n"
             "\n"
             "Returns:\n"
             "    The H1 norm of the function over its interval.")
        .def("colleague", &ComplexFunction::ColleagueMatrix,
             "The colleague matrix of the function.\n"
             "\n"
             "Returns:\n"
             "    The colleague matrix, whose eigenvalues give the roots of\n"
             "    the function.")
        .def("roots", &ComplexFunction::Roots,
             "The roots of the function.\n"
             "\n"
             "Returns:\n"
             "    A vector of the real roots of the function within its\n"
             "    interval.")
        .def("extrema", &ComplexFunction::Extrema,
             "The extrema of the function.\n"
             "\n"
             "Returns:\n"
             "    A vector of the points within the interval where the\n"
             "    derivative of the function vanishes.")
        .def("pow", &ComplexFunction::Power, nb::arg("n"),
             "Raise the function to an integer power.\n"
             "\n"
             "Args:\n"
             "    n: The power to which to raise the function.\n"
             "\n"
             "Returns:\n"
             "    A new function representing ``self`` raised to the power\n"
             "    ``n``.")
        .def("__pow__", &ComplexFunction::Power, nb::arg("n"),
             "Raise the function to an integer power. See :meth:`pow`.")
        .def("monomials", &ComplexFunction::Monomials,
             "The coefficients of the corresponding polynomial.\n"
             "\n"
             "Returns:\n"
             "    The coefficients of the polynomial (in the monomial\n"
             "    basis) equivalent to this Chebyshev series.")
        .def("product_matrix", &ComplexFunction::ProductMatrix, nb::arg("order"), nb::arg("rows") = -1,
             "The product matrix of the function.\n"
             "\n"
             "The product matrix is a matrix that, when multiplied by the\n"
             "vector of coefficients of another function, gives the\n"
             "coefficients of the product of the two functions.\n"
             "\n"
             "Args:\n"
             "    order: The order of the product matrix.\n"
             "    rows: The number of rows of the product matrix. Defaults\n"
             "        to a size large enough to hold the full product.\n"
             "\n"
             "Returns:\n"
             "    The product matrix of the function.");
}

NB_MODULE(cheby, m) {
    m.doc() = "Functions represented as Chebyshev series";
    m.attr("__version__") = CHEBYVERSION;
    m.def("RealConstant", &Constant<double>,
          "Construct a constant real function.\n"
          "\n"
          "Args:\n"
          "    xmin: The start of the interval.\n"
          "    xmax: The end of the interval.\n"
          "    c: The constant value of the function.\n"
          "\n"
          "Returns:\n"
          "    A :class:`RealFunction` equal to ``c`` everywhere on\n"
          "    ``[xmin, xmax]``.");
    m.def("ComplexConstant", &Constant<std::complex<double> >,
          "Construct a constant complex function.\n"
          "\n"
          "Args:\n"
          "    xmin: The start of the interval.\n"
          "    xmax: The end of the interval.\n"
          "    c: The constant value of the function.\n"
          "\n"
          "Returns:\n"
          "    A :class:`ComplexFunction` equal to ``c`` everywhere on\n"
          "    ``[xmin, xmax]``.");
    init_cheby(m);
}
