#ifndef _MATERIAL_MODEL_
#define _MATERIAL_MODEL_


#include "types.h"
#include "linalg.h"

template <class Particle>
class MaterialModelBase {
	
public:
	
	// should be contained in every material model:
	Mat computePF(Particle const & particle) const;	// couchy stress, i.e. partial_Psi/partial_F (Psi being the enerty)
	void postStepModification(Particle & particle) const; // modify particle at end of step. Used to e.g. introduce plasticity
	
	
public:
	
};



template <class Particle>
class MMFixedCorotated : public MaterialModelBase<Particle> {
	
public:
	// lame parameters
	real mu0;
	real lambda0;

public:
	
	MMFixedCorotated(real E = 1.0e5, real Nu = 0.3)
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
};

template <class Particle>
class MMSnow : public MMFixedCorotated<Particle> {

public:
	real hardening;
	
public:
	
	MMSnow(real E = 1.0e5, real Nu = 0.3, real hardening = 10)
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
};



#endif
