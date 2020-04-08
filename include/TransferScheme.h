#ifndef _TRANSFER_SCHEME_
#define _TRANSFER_SCHEME_
#include "types.h"
#include "gpu.h"

struct SimulationParameters {

  // global parameters for the actual MPM simulation.
  // (I separated these from the CLIOptions to keep them as lean as possible. The idea beeing that the
  // CLIOptions can then be used and extended to handle arbitrarily complex user input.)

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
    dx(1.0/(N)),
    dx_inv(1.0/dx)
  {}
};


// usage: one instance of this class for each particle
class TransferSchemeBase {

public:
  Veci range_begin;

  // ATTENTION: only to be called after p2g_prepare_particle() or g2p_prepare_particle()
  CUDA_HOSTDEV Veci get_range_begin() { return range_begin; }

  // the required methods for any TransferScheme should be listed here.
  // until that's the case, please refer to the MLS_APIC_Scheme as an example.

};

struct MLS_APIC_Particle : public ParticleBase {

  Mat C; // Affine momentum.
  real Jp; // Determinant of the deformation gradient.		// used for hardening of snow
  CUDA_HOSTDEV MLS_APIC_Particle(int type = 0, Vec x = Vec::Zero(), Vec v = Vec::Zero()) :
    ParticleBase(type, x, v),
    C(Mat::Zero()),
    Jp(1.0) {};
};


template <class InterpolationKernel>
class MLS_APIC_Scheme : public TransferSchemeBase {

public:
  Mat Dinv;
  Mat affine;

  Eigen::Matrix<real, 3, InterpolationKernel::size()> weights;

  template<class MaterialModel>
  __device__ void p2g_prepare_particle(MLS_APIC_Particle const & particle,
                            SimulationParameters const & par,
                            InterpolationKernel const & interpolationKernel,
                            MaterialModel const & materialModel) {
    // To be called for each particle in the particle-to-grid transfer
    weights = interpolationKernel.weights_per_direction(particle.x, par.dx_inv, range_begin);

    if (InterpolationKernel::d_is_const()) {
      Dinv = interpolationKernel.D_inv_const(par.dx_inv);
    } else {
      Dinv = interpolationKernel.D_inv(particle.x, range_begin, weights, par.dx);
    }

    Mat PF = materialModel.computePF(particle);

    // Cauchy stress times dt and inv_dx
    Mat stress = -Dinv * par.dt * materialModel.particleVolume * PF;

    affine = stress + materialModel.particleMass * particle.C;
  }

  __device__ Vec4 p2g_node_contribution(MLS_APIC_Particle const & particle,
                             Vec const & dist_part2node,
                             real particle_mass,
                             int i, int j, int k) {
    // The momentum & mass contribution the particle has on a single grid-point
    Vec momentum = particle.v * particle_mass;
    Vec momentum_affine_delta_pos = momentum + affine * dist_part2node;
    Vec4 momentum_mass(momentum_affine_delta_pos[0],
                       momentum_affine_delta_pos[1],
                       momentum_affine_delta_pos[2],
                       particle_mass);

    real weight = weights(0, i) * weights(1, j) * weights(2, k);

    return weight * momentum_mass;
  }

  __device__ void g2p_prepare_particle(MLS_APIC_Particle & particle,
                            SimulationParameters const & par,
                            InterpolationKernel const & interpolationKernel) {
    // To be called for each particle in the grid-to-particle transfer

    weights = interpolationKernel.weights_per_direction(particle.x, par.dx_inv, range_begin);

    if (InterpolationKernel::d_is_const()) {
      Dinv = interpolationKernel.D_inv_const(par.dx_inv);
    }
    else {
      Dinv = interpolationKernel.D_inv(particle.x, range_begin, weights, par.dx);
    }

    particle.C = Mat::Zero();
    particle.v = Vec::Zero();
  }

  __device__ void g2p_node_contribution(MLS_APIC_Particle & particle,
                             Vec const & dist_part2node,
                             Vec4 const & grid_node,
                             int i, int j, int k) {
    // adds, for each grid point, the respective contribution to the updated particle properties.
    real weight = weights(0, i) * weights(1, j) * weights(2, k);

    Vec v_grid = grid_node.head<3>();
    particle.v += weight * v_grid;

    //particle.C += (Dinv * v_grid) * (weight * dist_part2node).transpose();
    particle.C += (weight * v_grid) * (dist_part2node.transpose() * Dinv);
  }

  __device__ void g2p_finish_particle(MLS_APIC_Particle & particle,
                           SimulationParameters const & par) {
    // To be called directly after the particle-to-grid transfer.
    // For per-particle modifications that are specific to the transfer scheme.
    // Note: The MaterialModel class contains a similar method that will be called right after this one; to be used for material-specific per-particle modificaitons.

    // MLS-MPM F-update
    particle.F = (Mat::Identity() + par.dt * particle.C) * particle.F;
  }
};

#endif

