#ifndef CHEBY_FUNCTION_H
#define CHEBY_FUNCTION_H

#include <unsupported/Eigen/FFT>

#include "Eigen/Dense"

namespace cheby {

template <typename valueT, typename parameterT = double,
          typename indexT = std::size_t>
class Function {
   public:
    using Value = valueT;
    using Parameter = parameterT;
    using Index = indexT;
    using CoefVector = Eigen::Array<Value, Eigen::Dynamic, 1>;
    using ValueVector = Eigen::Array<Value, Eigen::Dynamic, 1>;
    using ParamVector = Eigen::Array<Parameter, Eigen::Dynamic, 1>;

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
};

}  // namespace cheby

#endif
