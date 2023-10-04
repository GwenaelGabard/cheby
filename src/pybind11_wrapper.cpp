#include "cheby/cheby.hpp"
#include "cheby_version.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace cheby;

void init_cheby(py::module &m){
    py::class_<Basis1D>(m, "Basis1D")
        .def(py::init<int, double, double>(), py::arg("order"), py::arg("start")=-1.0, py::arg("end")=1.0)
        .def("points1", &Basis1D::points1, py::arg("num_points")=-1)
        .def("points2", &Basis1D::points2, py::arg("num_points")=-1);
}


PYBIND11_MODULE(cheby, m) {
    m.doc() = "Functions represented as Chebyshev series";
    m.attr("__version__") = CHEBYVERSION;
    init_cheby(m);
}
