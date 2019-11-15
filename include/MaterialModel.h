#ifndef _MATERIAL_MODEL_
#define _MATERIAL_MODEL_


#include "types.h"
#include "linalg.h"

class MaterialModelBase {
	
public:
	//Mat computePF(Particle const & particle) const;	// should be contained in every material model
	
	
public:
	
};




class MMFixedCorotated : public MaterialModelBase {
	
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


class MMSnow : public MMFixedCorotated 
{
	
	real hardening;
	
public:
	
	MMSnow(real E = 1.0e5, real Nu = 0.3, real hardening = 10)
	    : MMFixedCorotated(E, Nu), hardening(hardening)
	{}
	
	
	Mat computePF(Particle const & particle) const
	{
		Mat R, S;
		polar_decomposition(particle.F, R, S);
		real e = std::exp(hardening * (1.0 - particle.Jp));
		real mu = mu0 * e;
		real lambda = lambda0 * e;
		real const & J = particle.Jp;
		return (2.0 * mu * (particle.F - R) * particle.F.transpose()) + lambda * ((J - 1.0) * J) * (particle.F.inverse().transpose());
	}
};



#endif
