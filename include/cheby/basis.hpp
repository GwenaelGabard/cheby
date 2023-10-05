#ifndef CHEBY_BASIS_H
#define CHEBY_BASIS_H

#include <cmath>

#include "Eigen/Dense"

namespace cheby {

template <typename valueT, typename parameterT = double,
          typename indexT = std::size_t>
class Basis {
   public:
    using Value = valueT;
    using Parameter = parameterT;
    using Index = indexT;
    using ValueVector = Eigen::Array<Value, Eigen::Dynamic, 1>;
    using ParamVector = Eigen::Array<Parameter, Eigen::Dynamic, 1>;

    static constexpr double rel_tol = 1.e-14;
    static constexpr Index tail_length = 8;

    Parameter xmin, xmax;
    Index order;

    Basis(const Index degree, const Parameter start = -1.0,
          const Parameter end = +1.0) {
        order = degree;
        xmin = start;
        xmax = end;
    };

    const ParamVector points1(int num_points = -1) const {
        if (num_points == 0) return ParamVector(0);
        if (num_points < 0) num_points = order + 1;
        ParamVector points(num_points);
        const Parameter a = EIGEN_PI / num_points;
        const Parameter b = EIGEN_PI / (2.0 * num_points);
        const Parameter c = (xmax + xmin) * 0.5;
        const Parameter d = (xmin - xmax) * 0.5;
        for (Index i = 0; i < num_points; ++i)
            points(i) = c + d * std::cos(a * i + b);
        return points;
    };

    const ParamVector points2(int num_points = -1) const {
        if ((num_points == 0) || (num_points == 1)) return ParamVector(0);
        if (num_points < 0) num_points = order + 1;
        ParamVector points(num_points);
        const Parameter a = EIGEN_PI / (num_points - 1);
        const Parameter b = (xmax + xmin) * 0.5;
        const Parameter c = (xmin - xmax) * 0.5;
        for (Index i = 0; i < num_points; ++i)
            points(i) = b + c * std::cos(a * i);
        return points;
    };
};

}  // namespace cheby

#endif
