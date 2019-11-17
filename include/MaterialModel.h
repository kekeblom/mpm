#ifndef _MATERIAL_MODEL_
#define _MATERIAL_MODEL_


#include "types.h"
#include "linalg.h"



inline real clamp(const real &number, const real &lower, const real &upper) {
  return std::max(std::min(number, upper), lower);
}




template <class Particle>
class MaterialModelBase {
	
	
public:
	
	// These should be implemented for every material model:
	Mat computePF(Particle const & particle) const;	// couchy stress, i.e. partial_Psi/partial_F (Psi being the enerty)
	void endOfStepMutation(Particle & particle) const; // modify particle at end of step. Used to e.g. introduce plasticity
	
	
public:
	
};



template <class Particle>
class MMFixedCorotated : public MaterialModelBase<Particle> {
	
	// "Neo-hookean based" model trying to avoid "inverted elements" (det(F) < 0)
	// [Stomakhin et al. 2012. Energetically Consistent Invertible Elasticity.]
	
public:
	// lame parameters
	real mu0;
	real lambda0;

public:
	
	MMFixedCorotated(real E = 1.0e4, real Nu = 0.3)
	{
		mu0 = E / (2 * (1 + Nu));
		lambda0 = E * Nu / ((1 + Nu) * (1 - 2 * Nu)); 
	}
	
	
	Mat computePF(Particle const & particle) const
	{
		Mat R, S;
		polar_decomposition(particle.F, R, S);
		real const & J = particle.Jp;
		return (2.0 * mu0 * (particle.F - R) * particle.F.transpose()) + lambda0 * ((J - 1.0) * J) * (particle.F.inverse().transpose());
	}
	
	void endOfStepMutation(Particle & particle) const
	{
		// nothing to do here
	}
};



template <class Particle>
class MMSnow : public MMFixedCorotated<Particle> {

	// adds plasticity and hardening
	
public:
	real hardening;
	
public:
	
	MMSnow(real E = 1.0e4, real Nu = 0.2, real hardening = 10)
	    : MMFixedCorotated<Particle>(E, Nu), 
	      hardening(hardening)
	{}
	
	
	Mat computePF(Particle const & particle) const
	{
		Mat R, S;
		polar_decomposition(particle.F, R, S);
		real e = std::exp(hardening * (1.0 - particle.Jp));
		real mu = this->mu0 * e;
		real lambda = this->lambda0 * e;
		real const & J = particle.Jp;
		return (2.0 * mu * (particle.F - R) * particle.F.transpose()) + lambda * ((J - 1.0) * J) * (particle.F.inverse().transpose());
	}
	
	
	void endOfStepMutation(Particle & particle) const
	{
		// plasticity
		Mat & F = particle.F;
		
		auto svd = F.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
		Mat sigma = svd.singularValues().asDiagonal();
		Mat U = svd.matrixU();
		Mat V = svd.matrixV();
		for (size_t i = 0; i < 2; i++) {
			sigma(i, i) = clamp(sigma(i, i), 1.0 - 2.5e-2, 1.0 + 7.5e-3);
		}
		
		real oldJ = F.determinant();
		F = U * sigma * V.transpose();
		
		real newJ = clamp(particle.Jp * oldJ / F.determinant(), 0.6, 20.0);
		particle.Jp = newJ;
		
	}
};



#endif
