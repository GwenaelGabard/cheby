#ifndef CHEBY_FUNCTION_H
#define CHEBY_FUNCTION_H

#include <unsupported/Eigen/FFT>

#include "Eigen/Dense"
#include "utils.hpp"

namespace cheby {

/// @brief Add two functions.
/// @param f1 The first function.
/// @param f2 The second function.
/// @return The sum of the two functions.
template <typename T1, typename T2, typename outT>
outT Add(const T1 &f1, const T2 &f2) {
    const typename T1::Index size1 = f1.coef.size();
    const typename T1::Index size2 = f2.coef.size();
    if (size1 > size2) {
        typename outT::CoefVector c = f1.coef;
        c.head(size2) += f2.coef;
        outT g(f1.xmin, f1.xmax, c);
        g.Trim();
        return (g);
    } else if (size2 > size1) {
        typename outT::CoefVector c = f2.coef;
        c.head(size1) += f1.coef;
        outT g(f1.xmin, f1.xmax, c);
        g.Trim();
        return (g);
    } else {
        typename outT::CoefVector c = f1.coef + f2.coef;
        outT g(f1.xmin, f1.xmax, c);
        g.Trim();
        return (g);
    }
}

/// @brief Subtract two functions.
/// @param f1 The first function.
/// @param f2 The second function.
/// @return The difference of the two functions.
template <typename T1, typename T2, typename outT>
outT Sub(const T1 &f1, const T2 &f2) {
    const typename T1::Index size1 = f1.coef.size();
    const typename T1::Index size2 = f2.coef.size();
    if (size1 > size2) {
        typename outT::CoefVector c = f1.coef;
        c.head(size2) -= f2.coef;
        outT g(f1.xmin, f1.xmax, c);
        g.Trim();
        return (g);
    } else if (size2 > size1) {
        typename outT::CoefVector c = -f2.coef;
        c.head(size1) += f1.coef;
        outT g(f1.xmin, f1.xmax, c);
        g.Trim();
        return (g);
    } else {
        typename outT::CoefVector c = f1.coef - f2.coef;
        outT g(f1.xmin, f1.xmax, c);
        g.Trim();
        return (g);
    }
}

/// @brief Multiply two functions.
/// @param f1 The first function.
/// @param f2 The second function.
/// @return The product of the two functions.
template <typename T1, typename T2, typename outT>
outT Multiply(const T1 &f1, const T2 &f2) {
    const typename T1::Index size1 = f1.coef.size();
    const typename T2::Index size2 = f2.coef.size();
    typename outT::CoefVector c = outT::CoefVector::Zero(size1 + size2);
    typename outT::Value p;
    for (int i1 = 0; i1 < size1; ++i1) {
        for (int i2 = 0; i2 < size2; ++i2) {
            p = f1.coef(i1) * f2.coef(i2);
            const auto j = abs(i1 - i2);
            c(i1 + i2) = c(i1 + i2) + p;
            c(j) = c(j) + p;
        }
    }
    c /= 2.0;
    outT g(f1.xmin, f1.xmax, c);
    g.Trim();
    return (g);
}

/// @brief A class for representing functions as Chebyshev series.
/// @tparam valueT The type of the values of the function.
/// @tparam parameterT The type of the parameter of the function.
/// @tparam indexT The type of the indices of the coefficients.
template <typename valueT, typename parameterT = double,
          typename indexT = std::size_t>
class Function {
   public:
    /// @brief The type of the values of the function.
    using Value = valueT;
    /// @brief The type of the parameter of the function.
    using Parameter = parameterT;
    /// @brief The type of the indices of the coefficients.
    using Index = indexT;
    /// @brief The type of a vector of coefficients.
    using CoefVector = Eigen::Array<Value, Eigen::Dynamic, 1>;
    /// @brief The type of a vector of values.
    using ValueVector = Eigen::Array<Value, Eigen::Dynamic, 1>;
    /// @brief The type of a matrix of values.
    using ValueMatrix = Eigen::Matrix<Value, Eigen::Dynamic, Eigen::Dynamic>;
    /// @brief The type of a vector of parameters.
    using ParamVector = Eigen::Array<Parameter, Eigen::Dynamic, 1>;
    /// @brief The type of the real part of a value.
    using RealPart = typename Eigen::NumTraits<Value>::Real;
    /// @brief The type of the Chebyshev basis.
    using Basis = cheby::Basis<Value, Parameter, Index>;

    static constexpr double rel_tol = 1.e-14;
    static constexpr Index tail_length = 8;

    /// @brief The vector of coefficients.
    CoefVector coef;
    /// @brief The start of the interval.
    Parameter xmin;
    /// @brief The end of the interval.
    Parameter xmax;

    /// @brief Default constructor for a function.
    Function() : xmin(-1.0), xmax(+1.0), coef(){};

    /// @brief Construct a Chebyshev representation of a function from a start,
    /// an end, and a vector of coefficients.
    /// @param start The start of the interval.
    /// @param end The end of the interval.
    /// @param c The vector of coefficients.
    Function(const Parameter start, const Parameter end, const CoefVector &c)
        : xmin(start), xmax(end), coef(c){};

    /// @brief Construct a Chebyshev representation of function from a start, an
    /// end, and the function.
    /// @param f The function to represent.
    /// @param start The start of the interval.
    /// @param end The end of the interval.
    Function(std::function<ValueVector(ParamVector)> f, const Parameter start,
             const Parameter end, const int N = -1) {
        xmin = start;
        xmax = end;
        if (N > 0) {
            ComputeCoef(f, xmin, xmax, N);
        } else {
            for (int k = 4; k <= 13; ++k) {
                ComputeCoef(f, xmin, xmax, pow(2, k));
                if (TailLength() >= tail_length) break;
            }
            Trim();
        }
    };

    /// @brief Calculate the coefficients of the Chebyshev representation of a
    /// function.
    /// @param f The function to represent.
    /// @param xmin The start of the interval.
    /// @param xmax The end of the interval.
    /// @param N The number of coefficients to calculate.
    void ComputeCoef(std::function<ValueVector(ParamVector)> f,
                     const Parameter xmin, const Parameter xmax, const int N) {
        auto xi = ParamVector::LinSpaced(N + 1, 0.0, EIGEN_PI).cos();
        Eigen::Matrix<Value, Eigen::Dynamic, 1> fn(2 * N);
        fn.head(N + 1) = f(xmin + (xmax - xmin) * (1.0 + xi) / 2.0);
        for (int n = 1; n < N; ++n) fn(2 * N - n) = fn(n);
        Eigen::FFT<double> fft;
        Eigen::VectorXcd fourier(2 * N);
        fft.fwd(fourier, fn);
        if constexpr (std::is_same<Value, double>::value)
            coef = fourier.head(N + 1).real() / N;
        else
            coef = fourier.head(N + 1) / N;
        coef[0] /= 2;
        coef[coef.size() - 1] /= 2;
    }

    /// @brief Get the Chebyshev basis of the function.
    /// @return The Chebyshev basis of the function.
    Basis GetBasis() const {
        Basis b(coef.size() - 1, xmin, xmax);
        return (b);
    }

    /// @brief Get the number of coefficients to trim from the end of the
    /// coefficient vector.
    /// @return The number of coefficients to trim from the end of the
    /// coefficient vector.
    Index TailLength() const {
        if (coef.size() == 0) return (0);
        Index n = coef.size() - 1;
        const double threshold = coef.abs().maxCoeff() * rel_tol;
        while ((abs(coef(n)) <= threshold) & (n > 0)) n--;
        return (coef.size() - n - 1);
    }

    /// @brief Trim the coefficient vector.
    void Trim() { coef.conservativeResize(coef.size() - TailLength()); }

    /// @brief Evaluate the function at a number of points.
    /// @param x The points at which to evaluate the function.
    /// @return The values of the function at the given points.
    ValueVector Eval(const ParamVector &x) const {
        if (coef.size() == 0) return (ValueVector::Zero(x.size()));
        if (coef.size() == 1) return (ValueVector::Constant(x.size(), coef(0)));
        const ParamVector xi = (x - xmin) / (xmax - xmin) * 2.0 - 1.0;
        if (coef.size() == 2) return (coef(0) + coef(1) * xi);
        const auto N = coef.size() - 1;
        ValueVector fk, a(x.size()), b(x.size());
        a.fill(coef(N));
        b.fill(0.0);
        for (int k = N - 1; k >= 1; --k) {
            fk = 2 * xi * a - b + coef(k);
            if (k > 1) {
                b = a;
                a = fk;
            }
        }
        return (xi * fk - a + coef(0));
    }

    /// @brief  Evaluate the derivative of the function at a number of points.
    /// @param x The points at which to evaluate the derivative of the function.
    /// @return The values of the derivative of the function at the given points.
    Function Derivative() const {
        if (coef.size() < 2) return (Function(xmin, xmax, CoefVector()));
        const Index N = coef.size() - 1;
        CoefVector g = CoefVector::Zero(N + 2);
        const Parameter constant = 4.0 / (xmax - xmin);
        double nn = N;
        for (Index n = N; n > 1; --n) {
            g(n - 1) = g(n + 1) + nn * coef(n) * constant;
            nn -= 1.0;
        }
        g(0) = (coef(1) * constant + g(2)) * 0.5;
        Function d(xmin, xmax, g);
        d.Trim();
        return (d);
    }

    /// @brief Compute the primitive (anti-derivative) of the function.
    /// @return The primitive of the function.
    Function Primitive() const {
        if (coef.size() == 0) return (Function(xmin, xmax, CoefVector()));
        if (coef.size() == 1) {
            CoefVector g(2);
            g(0) = 0.0;
            g(1) = coef(0) * (xmax - xmin) / 2.0;
            return (Function(xmin, xmax, g));
        }
        const Index N = coef.size() - 1;
        CoefVector g = CoefVector::Zero(N + 2);
        double nn = 2.0;
        for (Index n = 2; n <= N; ++n) {
            g(n - 1) -= coef(n) / 2.0 / (nn - 1.0);
            g(n + 1) += coef(n) / 2.0 / (nn + 1.0);
            nn += 1.0;
        }
        g(0) += coef(1);
        g(1) += coef(0);
        g(2) += coef(1) / 4.0;
        g *= (xmax - xmin) / 2.0;
        Function d(xmin, xmax, g);
        d.Trim();
        return (d);
    }

    /// @brief Compute the integral of the function over the whole interval.
    /// @return The integral of the function over the whole interval.
    Value Integral() const {
        if (coef.size() == 0) return (0.0);
        if (coef.size() == 1) return (coef(0) * (xmax - xmin));
        const Index N = coef.size() - 1;
        Value nn = 2.0;
        Value integral = coef(0);
        for (Index n = 2; n <= N; n += 2) {
            integral += coef(n) / (1.0 - nn * nn);
            nn += 2.0;
        }
        return (integral * (xmax - xmin));
    }

    /// @brief Compute the integral of the function over a subinterval.
    /// @param a The start of the subinterval.
    /// @param b The end of the subinterval.
    /// @return The integral of the function over the subinterval.
    Value Integral(const Parameter a, const Parameter b) const {
        ParamVector bounds(2);
        bounds(0) = a;
        bounds(1) = b;
        auto f = Primitive().Eval(bounds);
        return (f(1) - f(0));
    }

    /// @brief Construct another function with the real part of the function.
    /// @return The real part of the function.
    Function<RealPart, Parameter, Index> Real() const {
        return (Function<RealPart, Parameter, Index>(xmin, xmax, coef.real()));
    }

    /// @brief Construct another function with the imaginary part of the
    /// function.
    /// @return The imaginary part of the function.
    Function<RealPart, Parameter, Index> Imag() const {
        return (Function<RealPart, Parameter, Index>(xmin, xmax, coef.imag()));
    }

    /// @brief Construct another function with the complex conjugate of the
    /// function.
    /// @return The complex conjugate of the function.
    Function Conjugate() const {
        return (Function(xmin, xmax, coef.conjugate()));
    }

    /// @brief Compute the product matrix of the function.
    /// The product matrix is a matrix that, when multiplied by the vector of
    /// coefficients of another function, gives the coefficients of the product
    /// of the two functions.
    /// @param order The order of the product matrix.
    /// @param rows The number of rows of the product matrix.
    /// @return The product matrix of the function.
    ValueMatrix ProductMatrix(const Index order, const int rows = -1) const {
        const Index num_cols = order + 1;
        const Index num_coefs = coef.size();
        const Index num_rows = rows == -1 ? num_cols + num_coefs : rows;
        ValueMatrix matrix = ValueMatrix::Zero(num_rows, num_cols);
        for (int n = 0; n < num_cols; ++n) {
            for (int m = 0; m < num_coefs; ++m) {
                if (m + n < num_rows) matrix(m + n, n) += coef(m) / 2.0;
                if (abs(m - n) < num_rows)
                    matrix(abs(m - n), n) += coef(m) / 2.0;
            }
        }
        return (matrix);
    }

    /// @brief Compute the L2 norm of the function.
    /// @return The L2 norm of the function.
    RealPart NormL2() const {
        const auto product =
            Multiply<Function, Function, Function>(*this, Conjugate());
        return (sqrt(product.Real().Integral()));
    }

    /// @brief Compute the H1 norm of the function.
    /// @param alpha The weight of the derivative in the norm.
    /// @return The H1 norm of the function.
    RealPart NormH1(const RealPart alpha = 1.0) {
        const auto norm_f = NormL2();
        const auto norm_df = Derivative().NormL2();
        return (sqrt(norm_f * norm_f + alpha * norm_df * norm_df));
    }

    /// @brief Compute the colleague matrix of the function.
    /// @return The colleague matrix of the function.
    ValueMatrix ColleagueMatrix() const {
        const Index N = coef.size() - 1;
        ValueMatrix matrix = ValueMatrix::Zero(N, N);
        matrix(0, 1) = 1.0;
        for (Index n = 1; n < N - 1; ++n) {
            matrix(n, n - 1) = 0.5;
            matrix(n, n + 1) = 0.5;
        }
        matrix.row(N - 1) = -coef.head(N) / (2.0 * coef(N));
        matrix(N - 1, N - 2) += 0.5;
        return (matrix);
    }

    /// @brief Compute the roots of the function.
    /// @return A vector of roots of the function.
    ParamVector Roots() const {
        auto values = BalanceMatrix(ColleagueMatrix()).eigenvalues();
        ParamVector roots(values.size());
        Index j = 0;
        for (Index i = 0; i < values.size(); ++i) {
            const Parameter r = values(i).real();
            if ((r >= -1.0) & (r <= 1.0) & (values(i).imag() == 0.0)) {
                roots(j) = (r + 1.0) / 2.0 * (xmax - xmin) + xmin;
                j++;
            }
        }
        roots.conservativeResize(j);
        return (roots);
    }

    /// @brief Compute the extrema of the function.
    /// @return A vector of extrema of the function.
    ParamVector Extrema() const { return (Derivative().Roots()); }

    /// @brief Multiply two coefficient vectors.
    /// @param c1 The first coefficient vector.
    /// @param c2 The second coefficient vector.
    /// @param tol The tolerance used when trimming the result.
    /// @return The product of the two coefficient vectors.
    static CoefVector MultiplyCoef(const CoefVector &c1, const CoefVector &c2,
                                   const double tol = rel_tol) {
        const Index size1 = c1.size();
        const Index size2 = c2.size();
        CoefVector c = CoefVector::Zero(size1 + size2);
        Value p;
        for (int i1 = 0; i1 < size1; ++i1) {
            for (int i2 = 0; i2 < size2; ++i2) {
                p = c1(i1) * c2(i2);
                const auto j = abs(i1 - i2);
                c(i1 + i2) = c(i1 + i2) + p;
                c(j) = c(j) + p;
            }
        }
        c /= 2.0;
        if (c.size() > 0) {
            Index n = c.size() - 1;
            const double threshold = c.abs().maxCoeff() * tol;
            while ((abs(c(n)) <= threshold) & (n > 0)) n--;
            c.conservativeResize(n + 1);
        }
        return (c);
    }

    /// @brief Compute the power of the function.
    /// @param n The power to which to raise the function.
    /// @return The power of the function.
    Function Power(const Index n) const {
        if (n == 0) {
            CoefVector g(1);
            g(0) = 1.0;
            return (Function(xmin, xmax, g));
        }
        if (n == 1) return (*this);
        auto f2 = Multiply<Function, Function, Function>(*this, *this);
        if (n == 2) return (f2);
        const Index k = Index(log2(n)) + 1;
        std::vector<Function> f(k);
        f[0] = *this;
        f[1] = f2;
        for (Index i = 2; i < k; ++i)
            f[i] = Multiply<Function, Function, Function>(f[i - 1], f[i - 1]);
        Index p = 1 << (k - 1);
        Index m = n - p;
        Function fn = f[k - 1];
        for (int i = k - 2; i >= 0; --i) {
            p >>= 1;
            if (m >= p) {
                fn = Multiply<Function, Function, Function>(fn, f[i]);
                m -= p;
            }
        }
        return (fn);
    }
};

/// @brief Construct a constant function.
/// @param xmin The start of the interval.
/// @param xmax The end of the interval.
/// @param c The constant value of the function.
/// @return The constant function.
template <typename valueT, typename parameterT = double,
          typename indexT = std::size_t>
Function<valueT, parameterT, indexT> Constant(const parameterT xmin,
                                              const parameterT xmax,
                                              const valueT c) {
    typename Function<valueT, parameterT, indexT>::CoefVector coef(1);
    coef(0) = c;
    return (Function<valueT, parameterT, indexT>(xmin, xmax, coef));
};

}  // namespace cheby

#endif
