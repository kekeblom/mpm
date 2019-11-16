#ifndef _TRANSFER_SCHEME_
#define _TRANSFER_SCHEME_


#include "types.h"


class SimulationParameters {
	
public:
	// defining
	float dt;
	u32 N;
	// assumming cubic domain of size 1
	
	// redundant:
	real N_real;
	real dx;
	real dx_inv;
	
	SimulationParameters(float dt, u32 N) :
	    dt(dt),
	    N(N),
	    N_real(static_cast<real>(N)),
	    dx(1.0/(N-1)),
	    dx_inv(1.0/dx)
	{}
};







class MLS_APIC_Particle : public ParticleBase {
public:
	Mat F; // Deformation gradient.
	Mat C; // Affine momentum.
	real Jp; // Determinant of the deformation gradient.
	MLS_APIC_Particle(Vec x = Vec::Zero(), Vec v = Vec::Zero()) :
		ParticleBase(x, v),
		F(Mat::Identity()),
		C(Mat::Zero()),
		Jp(1.0)
	{}
};



// usage: one instance of this class for each particle
class TransferSchemeBase {
	
	
};


template <class InterpolationKernel>
class MLS_APIC_Scheme : public TransferSchemeBase {

public:
	
	Mat affine;
	Veci range_begin;
	Eigen::Matrix<real, 3, InterpolationKernel::size()> weights;
	
	
public:
	
	template<class MaterialModel>
	void p2g_prepare_particle(MLS_APIC_Particle const & particle, 
	                          SimulationParameters const & par, 
	                          real particle_volume,
	                          real particle_mass,
	                          InterpolationKernel const & interpolationKernel,
	                          MaterialModel const & materialModel)
	{
		
	    weights = interpolationKernel.weights_per_direction(particle.x, par.N_real, range_begin);
		
		Mat Dinv;
	    if (InterpolationKernel::d_is_const()) {
		    Dinv = interpolationKernel.D_inv(par.N_real);
	    }
	    else {
//		    Dinv = interpolationKernel.D_inv(particle.x, range_begin, weights, par.dx);
	    }
  
	    Mat PF = materialModel.computePF(particle);
	    
	    // Cauchy stress times dt and inv_dx
        Mat stress = -Dinv * par.dt * particle_volume * PF * particle.F.transpose();
  
        affine = stress + particle_mass * particle.C;
	}
	
	// ATTENTION: only to be called after p2g_prepare_particle()
	Veci get_range_begin()
	{
		return range_begin;
	}
	
	
	Vec4 p2g_node_contribution(MLS_APIC_Particle const & particle,
	                           Vec const & dist_part2node, 
	                           real particle_mass,
	                           int i, int j, int k)
	{
		
		Vec momentum = particle.v * particle_mass;
	    Vec momentum_affine_delta_pos = momentum + affine * dist_part2node;
	    Vec4 momentum_mass(momentum_affine_delta_pos(0),
						  momentum_affine_delta_pos(1),
						  momentum_affine_delta_pos(2),
						  particle_mass);
	    
	    real weight = weights(0, i) * weights(1, j) * weights(2, k);
		
		return weight * momentum_mass;
	}
	
};

#endif



