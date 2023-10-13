#ifndef CHEBY_UTILS_H
#define CHEBY_UTILS_H

#include "Eigen/Dense"

namespace cheby {

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> BalanceMatrix(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &A) {
    typename Eigen::NumTraits<T>::Real beta = 2.0;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B = A;
    bool converged = false;
    do {
        converged = true;
        for (Eigen::Index i = 0; i < A.rows(); ++i) {
            auto c = B.col(i).norm();
            auto r = B.row(i).norm();
            auto s = c * c + r * r;
            decltype(c) f = 1.0;
            while (c < r / beta) {
                c *= beta;
                r /= beta;
                f *= beta;
            }
            while (c >= r * beta) {
                c /= beta;
                r *= beta;
                f /= beta;
            }
            if ((c * c + r * r) < 0.95 * s) {
                converged = false;
                B.col(i) *= f;
                B.row(i) /= f;
            }
        }
    } while (!converged);
    return (B);
}

}  // namespace cheby

#endif
