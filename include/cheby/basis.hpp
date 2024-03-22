#ifndef CHEBY_BASIS_H
#define CHEBY_BASIS_H

#include <cmath>

#include "Eigen/Dense"

namespace cheby {

/// @brief A basis of Chebyshev polynomials of the first kind.
///
/// @tparam valueT The type of the values of the basis functions.
/// @tparam parameterT The type of the parameter of the basis functions.
/// @tparam indexT The type of the indices of the basis functions.
template <typename valueT, typename parameterT = double,
          typename indexT = std::size_t>
class Basis {
   public:
    /// @brief The type of the values of the basis.
    using Value = valueT;
    /// @brief The type of the parameters of the basis.
    using Parameter = parameterT;
    /// @brief The type of the indices of the basis.
    using Index = indexT;
    /// @brief The type of a vector of values.
    using ValueVector = Eigen::Array<Value, Eigen::Dynamic, 1>;
    /// @brief The type of a matrix of values.
    using ValueMatrix = Eigen::Array<Value, Eigen::Dynamic, Eigen::Dynamic>;
    /// @brief The type of a vector of parameters.
    using ParamVector = Eigen::Array<Parameter, Eigen::Dynamic, 1>;

    static constexpr double rel_tol = 1.e-14;
    static constexpr Index tail_length = 8;

    /// @brief The order of the basis.
    Index order;
    /// @brief The start of the interval.
    Parameter xmin;
    /// @brief The end of the interval.
    Parameter xmax;

    /// @brief Construct a basis.
    /// @param degree The order of the basis.
    /// @param start The start of the interval.
    /// @param end The end of the interval.
    Basis(const Index degree, const Parameter start = -1.0,
          const Parameter end = +1.0)
        : order(degree), xmin(start), xmax(end){};

    /// @brief Get the Chebyshev points of the first type.
    /// @param num_points The number of points to generate.
    /// @return A vector with the coordinates of the points.
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

    /// @brief Get the Chebyshev points of the second type.
    /// @param num_points The number of points to generate.
    /// @return A vector with the coordinates of the points.
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

    /// @brief Evaluate the Chebyshev polynomials of the first kind at the given
    /// points.
    /// @param x A vector of points at which to evaluate the polynomials.
    /// @return The matrix of polynomial values at the given points (each row is
    /// a point, each column is a polynomial).
    ValueMatrix Eval(const ParamVector x) const {
        const Index N = order + 1;
        ValueMatrix T(x.size(), N);
        // Optimisation: store 2\xi instead of \xi
        const ParamVector xi = (x - xmin) * (4.0 / (xmax - xmin)) - 2.0;
        T.col(0) = 1.0;
        if (N == 1) return T;
        T.col(1) = xi / 2.0;
        if (N == 2) return T;
        for (Index n = 2; n < N; ++n)
            T.col(n) = xi * T.col(n - 1) - T.col(n - 2);
        return T;
    };

    /// @brief Evaluate the derivatives of the Chebyshev polynomials of the first
    /// kind at the given points.
    /// @param x A vector of points at which to evaluate the polynomials.
    /// @param D The maximum order of the derivatives to evaluate.
    /// @return A vector of matrices of polynomial derivatives at the given
    /// points (each row is a point, each column is a polynomial).
    std::vector<ValueMatrix> Derivatives(const ParamVector x,
                                         const Index D) const {
        const Index N = order + 1;
        // Optimisation: store 2\xi instead of \xi
        const ParamVector xi = (x - xmin) * (4.0 / (xmax - xmin)) - 2.0;
        std::vector<ValueMatrix> T(D + 1);
        for (auto &t : T) t = ValueMatrix::Zero(x.size(), N);
        T[0].col(0) = 1.0;
        if (N == 1) return T;
        T[0].col(1) = xi / 2.0;
        if (D > 0) T[1].col(1) = 1.0;
        if (N == 2) return T;
        for (Index n = 2; n < N; ++n) {
            T[0].col(n) = xi * T[0].col(n - 1) - T[0].col(n - 2);
            for (Index d = 1; d <= D; ++d)
                T[d].col(n) = xi * T[d].col(n - 1) +
                              2 * d * T[d - 1].col(n - 1) - T[d].col(n - 2);
        }
        for (Index d = 1; d <= D; ++d) T[d] *= std::pow(2.0 / (xmax - xmin), d);
        return T;
    };

    /// @brief Provide the differentiation matrix for the Chebyshev polynomials
    /// of the first kind.
    /// @return The square differentiation matrix.
    const ValueMatrix DiffMatrix() const {
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

    /// @brief Provide the projection matrix for the Chebyshev polynomials of the
    /// first kind.
    /// @return The square projection matrix.
    const ValueMatrix ProjectionMatrix() const {
        const Index N = order + 1;
        ValueMatrix matrix = ValueMatrix::Zero(N, N);
        const Value jacobian = xmax - xmin;
        Index q = 0;
        for (Index m = 0; m < N; ++m) {
            for (Index n = q; n <= m; n += 2) {
                const Value sum = m + n;
                const Value dif = m - n;
                matrix(m, n) =
                    jacobian / (1.0 - sum * sum) + jacobian / (1.0 - dif * dif);
                matrix(n, m) = matrix(m, n);
            }
            q = 1 - q;
        }
        return matrix;
    };

    /// @brief Provide the matrix for the Dirichlet basis recombination.
    /// The recombined basis is zero at the end points, except for the first two
    /// basis functions. The first function is 1 and 0 at the start and end
    /// points, respectively. The second function is 0 and 1 at the start and end
    /// points, respectively.
    /// @return The square recombination matrix.
    const ValueMatrix DirichletMatrix() const {
        const Index N = order + 1;
        ValueMatrix matrix = ValueMatrix::Zero(N, N);
        for (Index i = 2; i < N; ++i) {
            matrix(i, i) = 0.5;
            matrix(i - 2, i) = -0.5;
        }
        matrix(0, 0) = 0.5;
        matrix(1, 0) = -0.5;
        matrix(0, 1) = 0.5;
        matrix(1, 1) = 0.5;
        return matrix;
    }

    /// @brief Provide the matrix to convert the Chebyshev coefficients into
    /// polynomial coefficients.
    /// @return A square matrix. The nth column contains the polynomial
    /// coefficients of the nth Chebyshev polynomial.
    const ValueMatrix MonomialMatrix() const {
        const Index N = order + 1;
        ValueMatrix matrix = ValueMatrix::Zero(N, N);
        matrix(0, 0) = 1.0;
        matrix(1, 1) = 1.0;
        for (Index i = 2; i < N; ++i) {
            matrix(Eigen::seq(1, N - 1), i) +=
                2.0 * matrix(Eigen::seq(0, N - 2), i - 1);
            matrix.col(i) -= matrix.col(i - 2);
        }
        return matrix;
    }
};

}  // namespace cheby

#endif
