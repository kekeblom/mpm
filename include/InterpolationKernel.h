#ifndef _INTERPOLATION_KERNEL_
#define _INTERPOLATION_KERNEL_


#include "types.h"

template<u32 N>
class InterpolationKernelBase {
	
public:
	
	
public:
	
	static constexpr u32 size() {return N;}
	
	Eigen::Matrix<real, Vec::SizeAtCompileTime, N> weights_per_direction(Vec const & x_particle, real dx_inv, Veci & range_begin) const;
	
};






class QuadraticInterpolationKernel : public InterpolationKernelBase<3> {
	
public:
	
public:
	
	Eigen::Matrix<real, Vec::SizeAtCompileTime, size()> weights_per_direction(Vec const & x_particle, real dx_inv, Veci & range_begin) const
	{
		Vec x_particle_gridpoints = x_particle * dx_inv;
		range_begin = (x_particle_gridpoints - Vec::Constant(0.5)).cast<int>();
		
		Vec fx = x_particle_gridpoints - range_begin.cast<real>();
		
		Eigen::Matrix<real, Vec::SizeAtCompileTime, size()> w;
		w.col(0) = Vec::Constant(0.5).array() * (Vec::Constant(1.5) - fx).array().square();
		w.col(1) = Vec::Constant(0.75).array() - (fx - Vec::Constant(1.0)).array().square();
		w.col(2) = Vec::Constant(0.5).array() * (fx - Vec::Constant(0.5)).array().square();
		
		return w;
	}


};













#endif
