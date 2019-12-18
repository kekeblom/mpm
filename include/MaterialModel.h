#ifndef _MATERIAL_MODEL_
#define _MATERIAL_MODEL_


#include "types.h"
#include "linalg.h"



inline real clamp(const real &number, const real &lower, const real &upper) {
  return std::max(std::min(number, upper), lower);
}

template <class Particle>
class MaterialModelBase {

public: //(private)

  real particleVolume;
  real particleMass;

  MaterialModelBase(real volume, real density)
    : particleVolume(volume)
  {
    particleMass = density * volume;
  }

  // These should be implemented for every material model:
  Mat computePF(Particle const & particle) const;	// couchy stress, i.e. partial_Psi/partial_F (Psi being the enerty)
  void endOfStepMutation(Particle & particle) const; // modify particle at end of step. Used to e.g. introduce plasticity

};



template <class Particle>
class MMFixedCorotated : public MaterialModelBase<Particle> {

  // "Neo-hookean based" model trying to avoid "inverted elements" (det(F) < 0)
  // [Stomakhin et al. 2012. Energetically Consistent Invertible Elasticity.]

public:
  // lame parameters
  real mu0;
  real lambda0;

  MMFixedCorotated(real volume, real density, real E, real Nu)
    : MaterialModelBase<Particle>(volume, density)
  {
    mu0 = E / (2 * (1 + Nu));
    lambda0 = E * Nu / ((1 + Nu) * (1 - 2 * Nu));
  }


  Mat computePF(Particle const & particle) const
  {
    Mat R, S;
    polar_decomposition(particle.F, R, S);
    real const & J = particle.Jp;
    return (2.0 * mu0 * (particle.F - R) * particle.F.transpose()) + lambda0 * ((J - 1.0) * J) * Mat::Identity();
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
  real plast_clamp_lower;
  real plast_clamp_higher;

  MMSnow(real volume,         // particle volume, defined by sampling density of the particles
         real density = 400,  //
         real E = 1.4e5,      // Youngs modulus
         real Nu = 0.2,       // Poisson ratio
         real hardening = 10, // snow "hardens" when under pressure. Set 0 to achieve Neo-hookean behaviour
         real plast_clamp_lower = 1.0-2.5e-2, real plast_clamp_higher = 1.0+7.5e-3) // These define the plasticity: the closer both are to 1, the more plasticity there is. Set to 0.0 and std::numeric_limits<real>::max() fully elastic material.
    : MMFixedCorotated<Particle>(volume, density, E, Nu),
      hardening(hardening),
      plast_clamp_lower(plast_clamp_lower),
      plast_clamp_higher(plast_clamp_higher)
  {}


  Mat computePF(Particle const & particle) const
  {
    Mat R, S;
    polar_decomposition(particle.F, R, S);
    real e = std::exp(hardening * (1.0 - particle.Jp));
    real mu = this->mu0 * e;
    real lambda = this->lambda0 * e;
    real const & J = particle.Jp;
    return (2.0 * mu * (particle.F - R) * particle.F.transpose()) + lambda * ((J - 1.0) * J) * Mat::Identity();
  }


  void endOfStepMutation(Particle & particle) const
  {
    // plasticity
    Mat & F = particle.F;

    auto svd = F.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat sigma = svd.singularValues().asDiagonal();
    Mat U = svd.matrixU();
    Mat V = svd.matrixV();
    for (size_t i = 0; i < 3; i++) {
      sigma(i, i) = clamp(sigma(i, i), plast_clamp_lower, plast_clamp_higher);
    }

    real oldJ = F.determinant();
    F = U * sigma * V.transpose();

    real newJ = clamp(particle.Jp * oldJ / F.determinant(), 0.6, 20.0);
    particle.Jp = newJ;

  }
};


template <class Particle>
class MMJelly : public MMFixedCorotated<Particle> {

  // adds plasticity and hardening

public:
  real hardening;

  MMJelly(real volume, real density = 1000, real E = 1.0e5, real Nu = 0.3, real hardening = 10)
    : MMFixedCorotated<Particle>(volume, density, E, Nu),
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
    return (2.0 * mu * (particle.F - R) * particle.F.transpose()) + lambda * ((J - 1.0) * J) * Mat::Identity();
  }

  void endOfStepMutation(Particle & particle) const
  {
    // plasticity
    Mat & F = particle.F;

    real oldJ = F.determinant();

    real newJ = clamp(particle.Jp * oldJ / F.determinant(), 0.6, 20.0);
    particle.Jp = newJ;
  }
};

#endif
