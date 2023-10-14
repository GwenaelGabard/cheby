#ifndef CHEBY_FUNCTION_H
#define CHEBY_FUNCTION_H

#include <unsupported/Eigen/FFT>

#include "Eigen/Dense"
#include "utils.hpp"

namespace cheby {

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

template <typename valueT, typename parameterT = double,
          typename indexT = std::size_t>
class Function {
   public:
    using Value = valueT;
    using Parameter = parameterT;
    using Index = indexT;
    using CoefVector = Eigen::Array<Value, Eigen::Dynamic, 1>;
    using ValueVector = Eigen::Array<Value, Eigen::Dynamic, 1>;
    using ValueMatrix = Eigen::Matrix<Value, Eigen::Dynamic, Eigen::Dynamic>;
    using ParamVector = Eigen::Array<Parameter, Eigen::Dynamic, 1>;
    using RealPart = typename Eigen::NumTraits<Value>::Real;

    static constexpr double rel_tol = 1.e-14;
    static constexpr Index tail_length = 8;

    CoefVector coef;
    Parameter xmin, xmax;

    Function() : coef(), xmin(-1.0), xmax(+1.0){};

    Function(const Parameter start, const Parameter end, const CoefVector &c) {
        xmin = start;
        xmax = end;
        coef = c;
    };

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

    const CoefVector &Coef() const { return (coef); }

    Index TailLength() const {
        if (coef.size() == 0) return (0);
        Index n = coef.size() - 1;
        const double threshold = coef.abs().maxCoeff() * rel_tol;
        while ((abs(coef(n)) <= threshold) & (n > 0)) n--;
        return (coef.size() - n - 1);
    }

    void Trim() { coef.conservativeResize(coef.size() - TailLength()); }

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

    Value Integral(const Parameter a, const Parameter b) const {
        ParamVector bounds(2);
        bounds(0) = a;
        bounds(1) = b;
        auto f = Primitive().Eval(bounds);
        return (f(1) - f(0));
    }

    Function<RealPart, Parameter, Index> Real() const {
        return (Function<RealPart, Parameter, Index>(xmin, xmax, coef.real()));
    }

    Function<RealPart, Parameter, Index> Imag() const {
        return (Function<RealPart, Parameter, Index>(xmin, xmax, coef.imag()));
    }

    Function Conjugate() const {
        return (Function(xmin, xmax, coef.conjugate()));
    }

    ValueMatrix ProductMatrix(const Index order) const {
        const Index N = order + 1;
        const Index M = coef.size();
        ValueMatrix matrix = ValueMatrix::Zero(M + N, N);
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
                matrix(m + n, n) += coef(m) / 2.0;
                matrix(abs(m - n), n) += coef(m) / 2.0;
            }
        }
        return (matrix);
    }

    RealPart NormL2() const {
        const auto product =
            Multiply<Function, Function, Function>(*this, Conjugate());
        return (sqrt(product.Real().Integral()));
    }

    RealPart NormH1(const RealPart alpha = 1.0) {
        const auto norm_f = NormL2();
        const auto norm_df = Derivative().NormL2();
        return (sqrt(norm_f * norm_f + alpha * norm_df * norm_df));
    }

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

    ParamVector Extrema() const { return (Derivative().Roots()); }
};

}  // namespace cheby

#endif
