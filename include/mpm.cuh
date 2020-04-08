#include <vector>
#include <iostream>
#include <algorithm>
#include <array>
#include <Eigen/Dense>
#include <assert.h>
#include <boost/filesystem.hpp>
#include <cmath>
#include <igl/readOBJ.h>
#include <igl/winding_number.h>

#include "options.h"
#include "linalg.h"
#include "utils.h"
#include "renderer.h"
#include "types.h"

#include "MaterialModel.cuh"
#include "InterpolationKernel.cuh"
#include "TransferScheme.h"

namespace fs = boost::filesystem;

using Particle = MLS_APIC_Particle;
using MaterialModel = MMSnow<Particle>;
using InterpolationKernel = QuadraticInterpolationKernel;
using TransferScheme = MLS_APIC_Scheme<InterpolationKernel>;

// defines a "physical" object to be simulated
struct SimObject {
  SimObject(MaterialModel model) : materialModel(model) {}
  // actual properties
  std::vector<Particle> particles;
  MaterialModel materialModel;
  // Objects may exist only for a limited time during the simulation
  real lifetime_begin = 0.0;
  real lifetime_end = std::numeric_limits<real>::max();
  bool isActive(real t) {
    return (t >= lifetime_begin && t < lifetime_end);
  }
};

// core class of the simulation - does all the heavy lifting.
class Simulation {

public:
  SimulationParameters par;
  u32 &N = par.N;
  int N2;
  int sizeof_grid;
  double t = 0.0;

  std::vector<SimObject> objects;

  int active_particle_count;
  MaterialModel* device_material_models;

  InterpolationKernel interpolationKernel; // how to interpolate from particles to grid to particle
  std::vector<MaterialModel> const &material_models;

private:
  std::vector<Particle> particles_all;  // used to accumulate all particles of the simulation, for simplified (but inefficient) access. For exporting stuff only.

  InterpolationKernel* device_interpolation_kernel;
  SimulationParameters* device_parameters;
  Vec4* device_grid;
  Particle* device_particles;
  Particle* host_particles;

public:
  Simulation(const CLIOptions &opts,
             InterpolationKernel const & interpolationKernel, std::vector<MaterialModel> const & material_models);
  ~Simulation();
  void initCuda();
  void advance();
  void syncDevice();

  void addObject(std::string const & filepath,  // obj file defining the shape of the object
                 int material_model,
                 real size,                     // size of the object, measured as the longest edge of the bounding box
                 Vec position,                  // position of the object within the scene (lowest corner of object)
                 Vec velocity,                  // initial velocity of the object
                 real lifetime_begin = 0.0,
                 real lifetime_end = std::numeric_limits<real>::max());

  size_t getFullParticleCount();
  std::vector<Particle> & getFullParticleList();
  std::vector<Particle> & getActiveParticleList();

private:
  void resetGrid();
  void particleToGridTransfer(std::vector<SimObject> & objects);
  void gridOperations();
  void particlesToDevice();
  void particlesToHost();
  void gridToParticleTransfer(std::vector<SimObject> & objects);

  void reallocateParticles();

  std::pair<Eigen::MatrixXf, Eigen::MatrixXi> loadMesh(const std::string& filepath, double size, Vec position);
  void addParticles(std::pair<Eigen::MatrixXf, Eigen::MatrixXi> const & mesh,
                    u32 particle_density,
                    Vec velocity,
                    std::vector<Particle> & particles,
                    u8 material_index);
};

