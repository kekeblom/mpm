#include <vector>
#include <iostream>
#include <algorithm>
#include <array>
#include <Eigen/Core>
#include <assert.h>
#include <boost/filesystem.hpp>
#include <boost/multi_array.hpp>
#include <omp.h>
#include <cmath>
#include <igl/readOBJ.h>
#include <igl/winding_number.h>

#include "options.h"
#include "linalg.h"
#include "utils.h"
#include "renderer.h"
#include "types.h"

#include "MaterialModel.h"
#include "InterpolationKernel.h"
#include "TransferScheme.h"

#include "particle_writer.h"
#include "mesh_builder.h"

namespace fs = boost::filesystem;

const unsigned int FrameRate = 240; // for export of data

const real Gravity = -9.81; // Gravity is, in good approximation, global ;)
const int NThreads = 8;


//-------------------------
// Setting up Simulation scenarios:
// to try different scenarios please scroll down to the main function.
// ------------------------


float get_random() {
  return float(rand()) / float(RAND_MAX);
}


// defines a "physical" object to be simulated
template<class MaterialModel, class Particle>
struct SimObject
{
  SimObject(MaterialModel materialModel) : materialModel(materialModel) {}
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
template<class MaterialModel, class TransferScheme, class Particle, class InterpolationKernel>
class Simulation {

public:
  SimulationParameters par;
  u32 & N = par.N;
  double t = 0.0;

  using SimObjectType = SimObject<MaterialModel, Particle>;
  std::vector<SimObjectType> objects; // Objects to be simulated

  boost::multi_array<Vec4, 3> grid; // Velocity x, y, z, mass

  InterpolationKernel interpolationKernel; // how to interpolate from particles to grid to particle

private:
  std::vector<Particle> particles_all;  // used to accumulate all particles of the simulation, for simplified (but inefficient) access. For exporting stuff only.

public:
  Simulation(const CLIOptions &opts,
             InterpolationKernel const & interpolationKernel)
    : par(opts.dt, opts.N),
      grid(boost::extents[par.N][par.N][par.N]),
      interpolationKernel(interpolationKernel)
  {}

  void resetGrid() {
    #pragma omp parallel for collapse(3) num_threads(NThreads)
    for (u32 i=0; i < N; i++) {
      for (u32 j=0; j < N; j++) {
        for (u32 k=0; k < N; k++) {
          grid[i][j][k](0) = 0;
          grid[i][j][k](1) = 0;
          grid[i][j][k](2) = 0;
          grid[i][j][k](3) = 0.0;
        }
      }
    }
  }

  // this is one iteration of the simulation
  void advance() {
    resetGrid();
    particleToGridTransfer(objects);
    gridOperations();
    gridToParticleTransfer(objects);
    t += par.dt;
  }

  template<class SimObjectType>
  void particleToGridTransfer(std::vector<SimObjectType> & objects) {
    // Particle-to-grid.
    for(SimObjectType & object : objects) { // loop through objects
      if(!object.isActive(t)) {continue;}
      auto & particles = object.particles;
      auto & materialModel = object.materialModel;
      #pragma omp parallel for num_threads(NThreads)
      for (u32 pi = 0; pi < particles.size(); ++pi) { // loop through particles
        Particle & particle = particles[pi];

        TransferScheme transferScheme;
        transferScheme.p2g_prepare_particle(particle,
                                            par,
                                            interpolationKernel,
                                            materialModel);

        Veci range_begin = transferScheme.get_range_begin();  // get start of range in grid that is influenced by the particle

        // handle particles that are completely outside of the domain
        bool out_of_range = false;
        for(int i = 0; i < 3; ++i) {
          if(range_begin(i)+int(interpolationKernel.size()) < 0 || range_begin(i) >= grid.shape()[i]) {
            out_of_range = true;
            break;
          }
        }
        if(out_of_range) {
          continue;
        }

        // bounds of particle-influence
        Vec dist_part2node;
        u32 i_begin = std::max(0, -range_begin(0));
        u32 j_begin = std::max(0, -range_begin(1));
        u32 k_begin = std::max(0, -range_begin(2));
        u32 i_end = std::min(interpolationKernel.size(), u32(grid.shape()[0]) - range_begin(0));
        u32 j_end = std::min(interpolationKernel.size(), u32(grid.shape()[1]) - range_begin(1));
        u32 k_end = std::min(interpolationKernel.size(), u32(grid.shape()[2]) - range_begin(2));

        // loop through relevant grid cells
        for(u32 i = i_begin; i < i_end; ++i) {
          u32 i_glob = range_begin(0) + i;
          dist_part2node(0) = i_glob * par.dx - particle.x(0);

          for(u32 j = j_begin; j < j_end; ++j) {
            u32 j_glob = range_begin(1) + j;
            dist_part2node(1) = j_glob * par.dx - particle.x(1);

            for(u32 k = k_begin; k < k_end; ++k) {
              u32 k_glob = range_begin(2) + k;
              dist_part2node(2) = k_glob * par.dx - particle.x(2);

              Vec4 node_contribution = transferScheme.p2g_node_contribution(particle, dist_part2node, materialModel.particleMass, i, j, k);

              for(int idx = 0; idx < 4; ++idx) {
                #pragma omp atomic
                grid[i_glob][j_glob][k_glob](idx) += node_contribution(idx);  // actual transfer
              }
            }
          }
        }
      }
    }
  }

  void gridOperations() {
    // Grid operations.
    #pragma omp parallel for collapse(3) num_threads(NThreads)
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        for (size_t k = 0; k < N; k++) {
          Vec4 &cell = grid[i][j][k];

          // boundary collisions
          if (cell[3] > 0.0) {
            cell /= cell[3];

            cell += par.dt * Vec4(0, Gravity, 0, 0);

            const real boundary = 0.05;

            const real x = real(i) / N;
            const real y = real(j) / N;
            const real z = real(k) / N;
            if (x < boundary || x > 1-boundary || y > 1-boundary ||
                z < boundary || z > 1-boundary) {
              cell = Vec4(0, 0, 0, cell[3]);
            }
            if (y < boundary) {
              cell[1] = std::max(real(0.0), cell[1]);
            }
          }
        }
      }
    }
  }

  template<class SimObjectType>
  void gridToParticleTransfer(std::vector<SimObjectType> & objects) {
    // Grid-to-particle.
    for(SimObjectType & object : objects) { // loop through objects
      if(!object.isActive(t)) {continue;}
      auto & particles = object.particles;
      auto & materialModel = object.materialModel;
      #pragma omp parallel for num_threads(NThreads)
      for (u32 pi = 0; pi < particles.size(); ++pi) { // loop through particles
        Particle & particle = particles[pi];

        TransferScheme transferScheme;
        transferScheme.g2p_prepare_particle(particle,
                                            par,
                                            interpolationKernel);

        Veci range_begin = transferScheme.get_range_begin(); // get start of range in grid that is influenced by the particle

        // handle particles that are completely outside of the domain
        bool out_of_range = false;
        for(int i = 0; i < 3; ++i) {
          if(range_begin(i)+int(interpolationKernel.size()) < 0 || range_begin(i) >= grid.shape()[i]) {
            out_of_range = true;
            break;
          }
        }
        if(out_of_range) {
          continue;
        }

        // bounds of particle-influence
        Vec dist_part2node;
        u32 i_begin = std::max(0, -range_begin(0));
        u32 j_begin = std::max(0, -range_begin(1));
        u32 k_begin = std::max(0, -range_begin(2));
        u32 i_end = std::min(interpolationKernel.size(), u32(grid.shape()[0]) - range_begin(0));
        u32 j_end = std::min(interpolationKernel.size(), u32(grid.shape()[1]) - range_begin(1));
        u32 k_end = std::min(interpolationKernel.size(), u32(grid.shape()[2]) - range_begin(2));

        // loop through relevant grid cells
        for(u32 i = i_begin; i < i_end; ++i) {
          u32 i_glob = range_begin(0) + i;
          dist_part2node(0) = i_glob * par.dx - particle.x(0);

          for(u32 j = j_begin; j < j_end; ++j) {
            u32 j_glob = range_begin(1) + j;
            dist_part2node(1) = j_glob * par.dx - particle.x(1);

            for(u32 k = k_begin; k < k_end; ++k) {
              u32 k_glob = range_begin(2) + k;
              dist_part2node(2) = k_glob * par.dx - particle.x(2);

              // actual transfer
              // velocity
              transferScheme.g2p_node_contribution(particle,
                                                   dist_part2node,
                                                   grid[i_glob][j_glob][k_glob],
                                                   i, j, k);
            }
          }
        }

        transferScheme.g2p_finish_particle(particle, par);

        // plasticity
        materialModel.endOfStepMutation(particle);

        // advection
        particle.x += par.dt * particle.v;
      }
    }
  }

  void addObject(std::string const & filepath,  // obj file defining the shape of the object
                 MaterialModel materialModel,
                 real size,                     // size of the object, measured as the longest edge of the bounding box
                 Vec position,                  // position of the object within the scene (lowest corner of object)
                 Vec velocity,                  // initial velocity of the object
                 real lifetime_begin = 0.0,
                 real lifetime_end = std::numeric_limits<real>::max())
  {
    auto mesh = loadMesh(filepath, size, position);

    objects.push_back(SimObjectType(materialModel));
    addParticles(mesh,
                 u32(1.0 / materialModel.particleVolume),
                 velocity,
                 objects.back().particles);
    objects.back().lifetime_begin = lifetime_begin;
    objects.back().lifetime_end = lifetime_end;
  }

  size_t getFullParticleCount()
  {
    size_t n = 0;
    for(auto & object : objects) {
      n += object.particles.size();
    }
    return n;
  }

  std::vector<Particle> & getFullParticleList()
  {
    particles_all.resize(0);
    for(auto & object : objects) {
      particles_all.insert(particles_all.end(), object.particles.begin(), object.particles.end());
    }
    return particles_all;
  }

  std::vector<Particle> & getActiveParticleList()
  {
    particles_all.resize(0);
    for(auto & object : objects) {
      if(!object.isActive(t)) {continue;}
      particles_all.insert(particles_all.end(), object.particles.begin(), object.particles.end());
    }
    return particles_all;
  }

private:

  std::pair<Eigen::MatrixXf, Eigen::MatrixXi> loadMesh(const std::string& filepath, double size, Vec position) {
    Eigen::MatrixXf V;
    Eigen::MatrixXi F;
    igl::readOBJ(filepath, V, F);

    Vec min = V.colwise().minCoeff();
    Vec max = V.colwise().maxCoeff();
    real length_max = (max - min).maxCoeff();
    real scale = size / length_max;

    V.array() *= scale;
    V.rowwise() += position.transpose() - scale * min.transpose();

    return std::make_pair(V, F);
  }

  void addParticles(std::pair<Eigen::MatrixXf, Eigen::MatrixXi> const & mesh,
                    u32 particle_density,
                    Vec velocity,
                    std::vector<Particle> & particles)
  {
    auto V = std::get<0>(mesh);
    auto F = std::get<1>(mesh);
    Eigen::MatrixXi W;
    const u32 BatchSize = 2048;
    Eigen::Matrix<float, BatchSize, 3> points;
    Vec x;

    int vertices = V.rows();
    double min_x = V.block(0, 0, vertices, 1).minCoeff();
    double max_x = V.block(0, 0, vertices, 1).maxCoeff();
    double min_y = V.block(0, 1, vertices, 1).minCoeff();
    double max_y = V.block(0, 1, vertices, 1).maxCoeff();
    double min_z = V.block(0, 2, vertices, 1).minCoeff();
    double max_z = V.block(0, 2, vertices, 1).maxCoeff();
    double range_x = max_x - min_x;
    double range_y = max_y - min_y;
    double range_z = max_z - min_z;

    double volume_bounding_box = range_x * range_y * range_z;
    u32 particle_count_target = particle_density * volume_bounding_box;

    u32 count_tot = 0;
    while (count_tot < particle_count_target) {
      for (u32 i=0; i < BatchSize; i++) {
        points(i, 0) = min_x + get_random() * range_x;
        points(i, 1) = min_y + get_random() * range_y;
        points(i, 2) = min_z + get_random() * range_z;
      }
      igl::winding_number(V, F, points, W);
      for (u32 i=0; i < BatchSize; i++) {
        count_tot++;
        if (W(i, 0) == 1) {
          x(0) = points(i, 0);
          x(1) = points(i, 1);
          x(2) = points(i, 2);
          particles.push_back(Particle(x, velocity));
        }
        if (count_tot == particle_count_target) {
          return;
        }
      }
    }
  }


};

int main(int argc, char *argv[]) {
  CLIOptions flags(argc, argv);


  using ParticleType = MLS_APIC_Particle;
  using MaterialModel = MMSnow<ParticleType>;
  using InterpolationKernel = QuadraticInterpolationKernel;
  using TransferScheme = MLS_APIC_Scheme<InterpolationKernel>;

  Simulation<MaterialModel,
             TransferScheme,
             ParticleType,
             InterpolationKernel> simulation(flags, InterpolationKernel());

  // "Scene selector"
  // usage: uncomment the desired scene below ;)
  // (also, set the correct path to the mesh directory)
  std::string meshes_dir = "../meshes/";

  ////////////////////////////
  // Scene: Snowman gets hit (by snowball)
  ////////////////////////////
  simulation.addObject(meshes_dir + "sphere.obj",
                       MaterialModel(1.0/flags.particle_count, 700, 1.4e5, 0.2, 10, 1.0-2.5e-2, 1.0+0.75e-2),  // soft, light snow
                       0.4,
                       Vec(0.3, 0.03, 0.3),
                       Vec(0.0,0.0,0.0));
  simulation.addObject(meshes_dir + "sphere.obj",
                       MaterialModel(1.0/flags.particle_count, 700, 1.4e5, 0.2, 10, 1.0-2.5e-2, 1.0+0.75e-2),  // soft, light snow
                       0.3,
                       Vec(0.35, 0.43, 0.35),
                       Vec(0.0,0.0,0.0));
  simulation.addObject(meshes_dir + "sphere.obj",
                       MaterialModel(1.0/flags.particle_count, 700, 1.4e5, 0.2, 10, 1.0-2.5e-2, 1.0+0.75e-2),  // soft, light snow
                       0.2,
                       Vec(0.4, 0.75, 0.4),
                       Vec(0.0,0.0,0.0));
  simulation.addObject(meshes_dir + "sphere.obj",
                       MaterialModel(1.0/flags.particle_count, 500, 1.4e5, 0.2, 10, 1.0-1.5e-2, 1.0+0.5e-2),  // soft, light snow
                       0.1,
                       Vec(0.05, 0.5, 0.46),
                       Vec(7.0,2.5,0.0),
                       0.5);

  ////////////////////////////
  // Scene: Rubber duck gets hit (by metal cube)
  ////////////////////////////
//  simulation.addObject(meshes_dir + "rubber_duck.obj",
//                       MaterialModel(1.0/flags.particle_count, 200, 1.4e5, 0.2, 0, 0.0, 1.0e30),
//                       0.4,
//                       Vec(0.3, 0.4, 0.3),
//                       Vec(0.0,0.0,0.0));
//  simulation.addObject(meshes_dir + "cube.obj",
//                       MaterialModel(1.0/flags.particle_count, 8000, 5e7, 0.45, 0, 0.0, 1.0e30), // "Fluid"????
//                       0.3,
//                       Vec(0.35, 0.5, 0.35),
//                       Vec(0.0,0.0,0.0),
//                       2.0);

  ////////////////////////////
  // Scene: Liquid bunny
  ////////////////////////////
//  simulation.addObject(meshes_dir + "stanford_bunny.obj",
//                       MaterialModel(1.0/flags.particle_count, 1000, 1.4e5, 0.45, 0, 0.975, 0.975),
//                       0.5,
//                       Vec(0.25, 0.4, 0.25),
//                       Vec(0.0,0.0,0.0),
//                       0.0);

  ////////////////////////////
  // Scene: Marshmallow duck and water
  ////////////////////////////
//  simulation.addObject(meshes_dir + "rubber_duck.obj",
//                       MaterialModel(1.0/flags.particle_count, 200, 1.4e5, 0.2, 0, 0.0, 1.0e30),
//                       0.4,
//                       Vec(0.3, 0.4, 0.3),
//                       Vec(0.0,0.0,0.0));
//  simulation.addObject(meshes_dir + "cube.obj",
//                       MaterialModel(1.0/flags.particle_count, 1000, 1.4e5, 0.45, 0, 0.975, 0.975), // "Fluid"????
//                       0.35,
//                       Vec(0.325, 0.5, 0.325),
//                       Vec(0.0,0.0,0.0),
//                       1.8);


  ////////////////////////////
  // General Examples with mesh from command line arguments
  ////////////////////////////
//  simulation.addObject(flags.load_mesh,
//                       MaterialModel(1.0/flags.particle_count, 400, 1.4e5, 0.2, 10, 1.0-1e-2, 1.0+3e-3),  // soft, light snow
//                       0.3,
//                       Vec(0.3, 0.6, 0.3),
//                       Vec(0.0,0.0,0.0));

//  simulation.addObject(flags.load_mesh,
//                       MaterialModel(1.0/flags.particle_count, 700, 1.4e5, 0.2, 10, 1.0-5e-2, 1.0+1.5e-2),  // harder, heavier snow
//                       0.2,
//                       Vec(0.35, 0.1, 0.3),
//                       Vec(0.0,5.0,0.0));

//  simulation.addObject(flags.load_mesh,
//                       MaterialModel(1.0/flags.particle_count, 400, 1.4e5, 0.2, 0, 0.0, 1.0e30),  // "Jelly", by abuse of the snow material model
//                       0.3,
//                       Vec(0.3, 0.6, 0.3),
//                       Vec(0.0,0.0,0.0));

//  simulation.addObject(flags.load_mesh,
//                       MaterialModel(1.0/flags.particle_count, 1000, 1.4e5, 0.45, 0, 0.97, 0.97), // "Fluid"????
//                       0.5,
//                       Vec(0.25, 0.4, 0.25),
//                       Vec(0.0,0.0,0.0));


  ParticleWriter writer;

  Renderer renderer(simulation.getFullParticleCount(), flags.save_dir);

  renderer.render(simulation.getActiveParticleList());

  MeshBuilder mesher(simulation.par, flags, flags.mesh_grid);

  bool save = flags.save_dir != "";
  if (save) {
    fs::create_directory(flags.save_dir);
    std::stringstream ss;
    ss << flags.save_dir << "/meshes";
    fs::create_directory(ss.str());
    ss.str("");
    ss.clear();
    ss << flags.save_dir << "/particles";
    fs::create_directory(ss.str());
  }

  u32 save_every = u32(1. / float(FrameRate) / flags.dt);
  u32 frame_id = 0;
  for (unsigned int i = 0; i < std::numeric_limits<unsigned int>::max(); i++) {
    std::cout << "Step " << i << "\r" << std::flush;
    simulation.advance();
    if(i%20 == 0) {
      renderer.render(simulation.getActiveParticleList());
    }
    if (save && (i % save_every) == 0) {
      std::stringstream ss;
      ss << flags.save_dir << "/meshes/mesh_" << std::setfill('0') << std::setw(5) << frame_id << ".obj";
      mesher.computeMesh(ss.str(), simulation.getActiveParticleList());
      ss.str("");
      ss.clear();
      ss << flags.save_dir << "/particles/particles_" << frame_id << ".bgeo";
      std::string filepath = ss.str();
      writer.writeParticles(filepath, simulation.getActiveParticleList());
      frame_id++;
    }
  }
}





