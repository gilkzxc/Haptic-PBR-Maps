#pragma once
#include <Eigen/Geometry>
#include <limits>


namespace radu{
namespace utils{

// mostly code modified from Sophus

inline Eigen::Matrix3d SophusSO3hat ( const Eigen::Vector3d & omega )
{
    Eigen::Matrix3d Omega;
    // clang-format off
    Omega <<
           double(0), -omega(2),  omega(1),
            omega(2), double(0), -omega(0),
           -omega(1),  omega(0), double(0);
    // clang-format on
    return Omega;
}

inline Eigen::Matrix3d SophusSO3expAndTheta ( const Eigen::Vector3d & omega, double * theta )
{
    const double const_eps = 1e-10;
    using std::abs;
    using std::cos;
    using std::sin;
    using std::sqrt;
    const double theta_sq = omega.squaredNorm();

    *theta = sqrt(theta_sq);
    const double half_theta = double(0.5) * (*theta);

    double imag_factor;
    double real_factor;
    if ((*theta) < const_eps) {
        const double theta_po4 = theta_sq * theta_sq;
        imag_factor = double(0.5) - double(1.0 / 48.0) * theta_sq +
                double(1.0 / 3840.0) * theta_po4;
        real_factor = double(1) - double(1.0 / 8.0) * theta_sq +
                double(1.0 / 384.0) * theta_po4;
    } else {
        const double sin_half_theta = sin(half_theta);
        imag_factor = sin_half_theta / (*theta);
        real_factor = cos(half_theta);
    }

    Eigen::Quaterniond q;
    q.w() = real_factor;
    q.x() = imag_factor * omega.x();
    q.y() = imag_factor * omega.y();
    q.z() = imag_factor * omega.z();

    return q.toRotationMatrix();
}

inline Eigen::Affine3d SophusExp ( const Eigen::Matrix<double,6,1> & a )
{
    const double const_eps = 1e-10;
    using std::cos;
    using std::sin;
    Eigen::Vector3d const omega = a.tail<3>();

    double theta;
    const Eigen::Matrix3d so3 = SophusSO3expAndTheta(omega, &theta);
    Eigen::Matrix3d const Omega = SophusSO3hat(omega);
    Eigen::Matrix3d const Omega_sq = Omega * Omega;
    Eigen::Matrix3d V;

    if (theta < const_eps ) {
        V = so3.matrix();
        // Note: That is an accurate expansion!
    } else {
        const double theta_sq = theta * theta;
        V = (Eigen::Matrix3d::Identity() +
             (double(1) - cos(theta)) / (theta_sq)*Omega +
             (theta - sin(theta)) / (theta_sq * theta) * Omega_sq);
    }
    Eigen::Affine3d res = Eigen::Affine3d::Identity();
    res.linear() = so3;
    res.translation() = V * a.head<3>();
    return res;
}

inline Eigen::Matrix<double,6,1> SophusLog ( const Eigen::Affine3d & e )
{
    if ( e.matrix().isApprox(Eigen::Matrix4d::Identity(),1e-6) )
        return Eigen::Matrix<double,6,1>::Zero();

    Eigen::Matrix<double,6,1> delta;
    const double theta = std::cos ( ( e.linear().trace() - 1 ) / 2 );
    const Eigen::Matrix3d Omega = theta / ( 2 * std::sin ( theta ) ) * ( e.linear() - e.linear().transpose() );
    delta.tail<3>() = Eigen::Vector3d ( Omega ( 2, 1 ), Omega ( 0, 2 ), Omega ( 1, 0 ) ); // vee operator.

    if ( std::abs ( theta ) < std::numeric_limits<double>::epsilon() )
    {
        const Eigen::Matrix3d V_inv = Eigen::Matrix3d::Identity() - 0.5 * Omega + 1. / 12. * ( Omega * Omega );
        delta.head<3>() = V_inv * e.translation();
    }
    else
    {
        const double half_theta = 0.5 * theta;
        const Eigen::Matrix3d V_inv = ( Eigen::Matrix3d::Identity() - 0.5 * Omega + ( 1 - theta * std::cos ( half_theta ) / ( 2 * std::sin ( half_theta ) ) ) / ( theta * theta ) * ( Omega * Omega ) );
        delta.head<3>() = V_inv * e.translation();
    }

    return delta;
}

inline Eigen::Affine3d interpolateSE3 ( const Eigen::Affine3d & a, const Eigen::Affine3d & b, const double & t )
{
    //const double inter_t = std::min(1.,std::max(0.,t));
    if ( t <= 0. ) return a;
    //if ( t >= 1. ) return b;
    return a * SophusExp(t * SophusLog(a.inverse()*b));
}
inline Eigen::Affine3d interpolateSE3AtId ( const Eigen::Affine3d & b, const double & t )
{
    if ( t <= 0. ) return Eigen::Affine3d::Identity();
    //if ( t >= 1. ) return b;
    return SophusExp(t * SophusLog(b));
}
} //namespace utils
} //namespace radu
