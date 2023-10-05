#ifndef CHEBY_CHEBY_H
#define CHEBY_CHEBY_H

#include "cheby/basis.hpp"
#include "cheby/function.hpp"

namespace cheby {

using Basis1D = Basis<double>;
using RealFunction = Function<double>;
using ComplexFunction = Function<std::complex<double> >;

}  // namespace cheby

#endif
