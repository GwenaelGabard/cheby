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

void init_cheby(nb::module_ &m) {
    nb::class_<Basis1D>(m, "Basis1D")
        .def(nb::init<int, double, double>(), nb::arg("order"),
             nb::arg("start") = -1.0, nb::arg("end") = 1.0)
        .def_ro("order", &Basis1D::order)
        .def_ro("start", &Basis1D::xmin)
        .def_ro("end", &Basis1D::xmax)
        .def("points1", &Basis1D::Points1, nb::arg("num_points") = -1)
        .def("points2", &Basis1D::Points2, nb::arg("num_points") = -1)
        .def("eval", &Basis1D::Eval)
        .def("derivatives", &Basis1D::Derivatives)
        .def("dirichlet", &Basis1D::DirichletMatrix)
        .def("neumann", &Basis1D::NeumannMatrix)
        .def("diff_matrix", &Basis1D::DiffMatrix)
        .def("projection_matrix", &Basis1D::ProjectionMatrix)
        .def("monomial_matrix", &Basis1D::MonomialMatrix);

    nb::class_<RealFunction>(m, "RealFunction")
        .def(nb::init<std::function<RealFunction::ValueVector(
                          RealFunction::ParamVector)>,
                      double, double, int>(),
             nb::arg("f"), nb::arg("start"), nb::arg("end"), nb::arg("N") = -1)
        .def(nb::init<double, double, RealFunction::CoefVector>(),
             nb::arg("start"), nb::arg("end"), nb::arg("coef"))
        .def("__call__", &RealFunction::Eval)
        .def("__add__", &Add<RealFunction, RealFunction, RealFunction>)
        .def("__add__", &Add<RealFunction, ComplexFunction, ComplexFunction>)
        .def("__sub__", &Sub<RealFunction, RealFunction, RealFunction>)
        .def("__sub__", &Sub<RealFunction, ComplexFunction, ComplexFunction>)
        .def("__mul__", &Multiply<RealFunction, RealFunction, RealFunction>)
        .def("__mul__",
             &Multiply<RealFunction, ComplexFunction, ComplexFunction>)
        .def_ro("start", &RealFunction::xmin)
        .def_ro("end", &RealFunction::xmax)
        .def_ro("coef", &RealFunction::coef)
        .def("basis", &RealFunction::GetBasis)
        .def("tail_length", &RealFunction::TailLength)
        .def("trim", &RealFunction::Trim)
        .def("derivative", &RealFunction::Derivative)
        .def("primitive", &RealFunction::Primitive)
        .def("integral",
             nb::overload_cast<>(&RealFunction::Integral, nb::const_))
        .def("integral", nb::overload_cast<const RealFunction::Parameter,
                                           const RealFunction::Parameter>(
                             &RealFunction::Integral, nb::const_))
        .def("real", &RealFunction::Real)
        .def("imag", &RealFunction::Imag)
        .def("conj", &RealFunction::Conjugate)
        .def("norm_L2", &RealFunction::NormL2)
        .def("norm_H1", &RealFunction::NormH1, nb::arg("alpha") = 1.0)
        .def("colleague", &RealFunction::ColleagueMatrix)
        .def("roots", &RealFunction::Roots)
        .def("extrema", &RealFunction::Extrema)
        .def("pow", &RealFunction::Power)
        .def("__pow__", &RealFunction::Power)
        .def("monomials", &RealFunction::Monomials)
        .def("inverse", &RealFunction::Inverse)
        .def("product_matrix", &RealFunction::ProductMatrix, nb::arg("order"),
             nb::arg("rows") = -1);

    nb::class_<ComplexFunction>(m, "ComplexFunction")
        .def(nb::init<std::function<ComplexFunction::ValueVector(
                          ComplexFunction::ParamVector)>,
                      double, double, int>(),
             nb::arg("f"), nb::arg("start"), nb::arg("end"), nb::arg("N") = -1)
        .def(nb::init<double, double, ComplexFunction::CoefVector>(),
             nb::arg("start"), nb::arg("end"), nb::arg("coef"))
        .def("__call__", &ComplexFunction::Eval)
        .def("__add__", &Add<ComplexFunction, RealFunction, ComplexFunction>)
        .def("__add__", &Add<ComplexFunction, ComplexFunction, ComplexFunction>)
        .def("__sub__", &Sub<ComplexFunction, RealFunction, ComplexFunction>)
        .def("__sub__", &Sub<ComplexFunction, ComplexFunction, ComplexFunction>)
        .def("__mul__",
             &Multiply<ComplexFunction, ComplexFunction, ComplexFunction>)
        .def("__mul__",
             &Multiply<ComplexFunction, RealFunction, ComplexFunction>)
        .def_ro("start", &ComplexFunction::xmin)
        .def_ro("end", &ComplexFunction::xmax)
        .def_ro("coef", &ComplexFunction::coef)
        .def("basis", &ComplexFunction::GetBasis)
        .def("tail_length", &ComplexFunction::TailLength)
        .def("trim", &ComplexFunction::Trim)
        .def("derivative", &ComplexFunction::Derivative)
        .def("primitive", &ComplexFunction::Primitive)
        .def("integral",
             nb::overload_cast<>(&ComplexFunction::Integral, nb::const_))
        .def("integral", nb::overload_cast<const ComplexFunction::Parameter,
                                           const ComplexFunction::Parameter>(
                             &ComplexFunction::Integral, nb::const_))
        .def("real", &ComplexFunction::Real)
        .def("imag", &ComplexFunction::Imag)
        .def("conj", &ComplexFunction::Conjugate)
        .def("norm_L2", &ComplexFunction::NormL2)
        .def("norm_H1", &ComplexFunction::NormH1, nb::arg("alpha") = 1.0)
        .def("colleague", &ComplexFunction::ColleagueMatrix)
        .def("roots", &ComplexFunction::Roots)
        .def("extrema", &ComplexFunction::Extrema)
        .def("pow", &ComplexFunction::Power)
        .def("__pow__", &ComplexFunction::Power)
        .def("monomials", &ComplexFunction::Monomials)
        .def("product_matrix", &ComplexFunction::ProductMatrix, nb::arg("order"),
             nb::arg("rows") = -1);
}

NB_MODULE(cheby, m) {
    m.doc() = "Functions represented as Chebyshev series";
    m.attr("__version__") = CHEBYVERSION;
    m.def("RealConstant", &Constant<double>);
    m.def("ComplexConstant", &Constant<std::complex<double> >);
    init_cheby(m);
}
