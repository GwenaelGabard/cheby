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
    using ValueMatrix = Eigen::Array<Value, Eigen::Dynamic, Eigen::Dynamic>;
    using ParamVector = Eigen::Array<Parameter, Eigen::Dynamic, 1>;

    static constexpr double rel_tol = 1.e-14;
    static constexpr Index tail_length = 8;

    Index order;
    Parameter xmin, xmax;

    Basis(const Index degree, const Parameter start = -1.0,
          const Parameter end = +1.0)
        : order(degree), xmin(start), xmax(end){};

    const ParamVector Points1(int num_points = -1) const {
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

    const ParamVector Points2(int num_points = -1) const {
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

    ValueMatrix Eval(const ParamVector x) const {
        const Index N = order + 1;
        ValueMatrix T(x.size(), N);
        const ParamVector xi = (x - xmin) * (4.0 / (xmax - xmin)) - 2.0;
        T.col(0) = 1.0;
        if (N == 1) return T;
        T.col(1) = xi / 2.0;
        if (N == 2) return T;
        for (Index n = 2; n < N; ++n)
            T.col(n) = xi * T.col(n - 1) - T.col(n - 2);
        return T;
    };

    const ValueMatrix Derivative() const {
        const Index N = order + 1;
        ValueMatrix D = ValueMatrix::Zero(N, N);
        if (N == 1) return D;
        const Value jacobian = 4.0 / (xmax - xmin);
        D(0, 1) = jacobian * 0.5;
        if (N == 2) return (D);
        D(1, 2) = jacobian * 2.0;
        for (Index n = 3; n < N; ++n) {
            const Value nn = static_cast<Value>(n);
            D(n - 1, n) = jacobian * nn;
            D.col(n) += nn / (nn - 2.0) * D.col(n - 2);
        }
        return D;
    };
};

}  // namespace cheby

#endif
