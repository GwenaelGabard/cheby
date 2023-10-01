#ifndef CHEBY_Basis_H
#define CHEBY_Basis_H

#include <vector>
#include <cmath>
#include "Eigen/Dense"

namespace cheby {

template <typename valueT, typename parameterT=double, typename indexT=std::size_t>
class Basis {
public:
    using Value = valueT;
    using Parameter = parameterT;
    using Index = indexT;
    using ValVector = Eigen::Array<Value,Eigen::Dynamic,1>;
    using ParamVector = Eigen::Array<Parameter,Eigen::Dynamic,1>;
    
    static constexpr double rel_tol = 1.e-14;
    static constexpr Index tail_length = 8;
    
    Parameter xmin, xmax;
    Index order;

    Basis(const Index degree, const Parameter start=-1.0, const Parameter end=+1.0) {
        order = degree;
        xmin = start;
        xmax = end;
    };

    const ParamVector points2(Index num_points=0) const {
        if (num_points == 0)
            num_points = order + 1;
        ParamVector points = (Eigen::VectorXd::LinSpaced(num_points, 0, static_cast<double>(num_points)).array()*EIGEN_PI / num_points).cos();
        points = (1.0 - points.array()) * 0.5 * (xmax - xmin) + xmin;
        return points;
    };
};


}

#endif
