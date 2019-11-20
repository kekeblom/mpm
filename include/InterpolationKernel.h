#ifndef _INTERPOLATION_KERNEL_
#define _INTERPOLATION_KERNEL_


#include "types.h"

template<u32 N, bool D_is_const = false>
class InterpolationKernelBase {

public:


public:


  static constexpr u32 size() {return N;}
  static constexpr bool d_is_const() {return D_is_const;}

  // The following should be implemented for every material model:
  Eigen::Matrix<real, 3, N> weights_per_direction(Vec const & x_particle, real dx_inv, Veci & range_begin) const;

  Eigen::Matrix<real, 3, 3> D_inv(Vec const & x_particle, Veci const & range_begin, Eigen::Matrix<real, 3, N> const & weights, real dx) const
  {
    // required for APIC.
    // default definition -> use if performance is not an issue.
    // For some interpolation schemes this simplifies dramatically.
    // If d_inv is constant, implement simplified version below.

    Eigen::Matrix<real, 3, 3> D = Eigen::Matrix<real, 3, 3>::Zero();

    Vec diff_part2node;
    for(u32 i = 0; i < size(); ++i) {
      u32 i_glob = range_begin(0) + i;
      diff_part2node(0) = i_glob * dx - x_particle(0);

      for(u32 j = 0; j < size(); ++j) {
        u32 j_glob = range_begin(1) + j;
        diff_part2node(1) = j_glob * dx - x_particle(1);

        for(u32 k = 0; k < size(); ++k) {
          u32 k_glob = range_begin(2) + k;
          diff_part2node(2) = k_glob * dx - x_particle(2);

          real weight = weights(0, i) * weights(1, j) * weights(2, k);

          D += (weight * diff_part2node) * diff_part2node.transpose();
        }
      }
    }

    return(D.inverse());
  }
  // if d_inv is constant, implement this simplified version
  Mat D_inv_const(real dx_inv) const;

};






class QuadraticInterpolationKernel : public InterpolationKernelBase<3, true> {

public:

public:

  Eigen::Matrix<real, 3, size()> weights_per_direction(Vec const & x_particle, real dx_inv, Veci & range_begin) const
  {
    Vec x_particle_gridpoints = x_particle * dx_inv;
    range_begin = (x_particle_gridpoints - Vec::Constant(0.5)).cast<int>();

    Vec fx = x_particle_gridpoints - range_begin.cast<real>();

    Eigen::Matrix<real, 3, size()> w;
    w.col(0) = Vec::Constant(0.5).array() * (Vec::Constant(1.5) - fx).array().square();
    w.col(1) = Vec::Constant(0.75).array() - (fx - Vec::Constant(1.0)).array().square();
    w.col(2) = Vec::Constant(0.5).array() * (fx - Vec::Constant(0.5)).array().square();

    return w;
  }


  Mat D_inv_const(real dx_inv) const
  {
    return Mat::Identity() * 4.0 * dx_inv * dx_inv;
  }

};













#endif
