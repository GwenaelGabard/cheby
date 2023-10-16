#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cheby/cheby.hpp"
#include "cheby_version.hpp"

namespace py = pybind11;
using namespace cheby;

void init_cheby(py::module &m) {
    py::class_<Basis1D>(m, "Basis1D")
        .def(py::init<int, double, double>(), py::arg("order"),
             py::arg("start") = -1.0, py::arg("end") = 1.0)
        .def("points1", &Basis1D::Points1, py::arg("num_points") = -1)
        .def("points2", &Basis1D::Points2, py::arg("num_points") = -1)
        .def("eval", &Basis1D::Eval)
        .def("derivatives", &Basis1D::Derivatives)
        .def("diff_matrix", &Basis1D::DiffMatrix);

    py::class_<RealFunction>(m, "RealFunction")
        .def(py::init<std::function<RealFunction::ValueVector(
                          RealFunction::ParamVector)>,
                      double, double, int>(),
             py::arg("f"), py::arg("start"), py::arg("end"), py::arg("N") = -1)
        .def("__call__", &RealFunction::Eval)
        .def("__add__", &Add<RealFunction, RealFunction, RealFunction>)
        .def("__add__", &Add<RealFunction, ComplexFunction, ComplexFunction>)
        .def("__sub__", &Sub<RealFunction, RealFunction, RealFunction>)
        .def("__sub__", &Sub<RealFunction, ComplexFunction, ComplexFunction>)
        .def("__mul__", &Multiply<RealFunction, RealFunction, RealFunction>)
        .def("__mul__",
             &Multiply<RealFunction, ComplexFunction, ComplexFunction>)
        .def("coef", &RealFunction::Coef)
        .def("tail_length", &RealFunction::TailLength)
        .def("trim", &RealFunction::Trim)
        .def("derivative", &RealFunction::Derivative)
        .def("primitive", &RealFunction::Primitive)
        .def("integral",
             py::overload_cast<>(&RealFunction::Integral, py::const_))
        .def("integral", py::overload_cast<const RealFunction::Parameter,
                                           const RealFunction::Parameter>(
                             &RealFunction::Integral, py::const_))
        .def("real", &RealFunction::Real)
        .def("imag", &RealFunction::Imag)
        .def("conj", &RealFunction::Conjugate)
        .def("norm_L2", &RealFunction::NormL2)
        .def("norm_H1", &RealFunction::NormH1, py::arg("alpha") = 1.0)
        .def("colleague", &RealFunction::ColleagueMatrix)
        .def("roots", &RealFunction::Roots)
        .def("extrema", &RealFunction::Extrema)
        .def("pow", &RealFunction::Power)
        .def("product_matrix", &RealFunction::ProductMatrix);

    py::class_<ComplexFunction>(m, "ComplexFunction")
        .def(py::init<std::function<ComplexFunction::ValueVector(
                          ComplexFunction::ParamVector)>,
                      double, double, int>(),
             py::arg("f"), py::arg("start"), py::arg("end"), py::arg("N") = -1)
        .def("__call__", &ComplexFunction::Eval)
        .def("__add__", &Add<ComplexFunction, RealFunction, ComplexFunction>)
        .def("__add__", &Add<ComplexFunction, ComplexFunction, ComplexFunction>)
        .def("__sub__", &Sub<ComplexFunction, RealFunction, ComplexFunction>)
        .def("__sub__", &Sub<ComplexFunction, ComplexFunction, ComplexFunction>)
        .def("__mul__",
             &Multiply<ComplexFunction, ComplexFunction, ComplexFunction>)
        .def("__mul__",
             &Multiply<ComplexFunction, RealFunction, ComplexFunction>)
        .def("coef", &ComplexFunction::Coef)
        .def("tail_length", &ComplexFunction::TailLength)
        .def("trim", &ComplexFunction::Trim)
        .def("derivative", &ComplexFunction::Derivative)
        .def("primitive", &ComplexFunction::Primitive)
        .def("integral",
             py::overload_cast<>(&ComplexFunction::Integral, py::const_))
        .def("integral", py::overload_cast<const ComplexFunction::Parameter,
                                           const ComplexFunction::Parameter>(
                             &ComplexFunction::Integral, py::const_))
        .def("real", &ComplexFunction::Real)
        .def("imag", &ComplexFunction::Imag)
        .def("conj", &ComplexFunction::Conjugate)
        .def("norm_L2", &ComplexFunction::NormL2)
        .def("norm_H1", &ComplexFunction::NormH1, py::arg("alpha") = 1.0)
        .def("colleague", &ComplexFunction::ColleagueMatrix)
        .def("roots", &ComplexFunction::Roots)
        .def("extrema", &ComplexFunction::Extrema)
        .def("pow", &ComplexFunction::Power)
        .def("product_matrix", &ComplexFunction::ProductMatrix);
}

PYBIND11_MODULE(cheby, m) {
    m.doc() = "Functions represented as Chebyshev series";
    m.attr("__version__") = CHEBYVERSION;
    init_cheby(m);
}
