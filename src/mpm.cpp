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

const unsigned int FrameRate = 120;

const real ParticleMass = 1.0;
const real ParticleVolume = 1.0;
const real Gravity = -9.81;
const int NThreads = 8;

float get_random() {
  return float(rand()) / float(RAND_MAX);
}

template<class MaterialModel, class InterpolationKernel, class TransferScheme, class Particle>
class Simulation {
  private:
  std::pair<Eigen::MatrixXf, Eigen::MatrixXi> loadMesh(const std::string& filepath) {
    Eigen::MatrixXf V;
    Eigen::MatrixXi F;
    igl::readOBJ(filepath, V, F);
    float min = V.minCoeff();
    V = (V.array() - min).matrix();
    float max = V.maxCoeff() * 3;
    V = V / max;
    V = (0.3 + V.array()).matrix();
    return std::make_pair(V, F);
  };

  void addParticles(const std::string& filepath) {
    u32 count = 0;
    auto mesh = loadMesh(filepath);
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

    while (count < particle_count_target) {
      for (u32 i=0; i < BatchSize; i++) {
        points(i, 0) = min_x + get_random() * range_x;
        points(i, 1) = min_y + get_random() * range_y;
        points(i, 2) = min_z + get_random() * range_z;
      }
      igl::winding_number(V, F, points, W);
      for (u32 i=0; i < BatchSize; i++) {
        if (W(i, 0) == 1) {
          x(0) = points(i, 0);
          x(1) = points(i, 1);
          x(2) = points(i, 2);
          particles.push_back(Particle(x));
          count++;
          if (count == particle_count_target) {
            return;
          }
        }
      }
    }
  }

  public:
    SimulationParameters par;
    u32 & N = par.N;
    const u32 particle_count_target;

    std::vector<Particle> particles;
    boost::multi_array<Vec4, 3> grid; // Velocity x, y, z, mass

    MaterialModel materialModel;
    InterpolationKernel interpolationKernel;

    Simulation(const CLIOptions &opts,
             MaterialModel const & materialModel,
             InterpolationKernel const & interpolationKernel,
             TransferScheme const & transferScheme_dummy,
             Particle const & particle_dummy)
    : par(opts.dt, opts.N),
      particle_count_target(opts.particle_count),
      grid(boost::extents[par.N][par.N][par.N]),
      materialModel(materialModel),
      interpolationKernel(interpolationKernel) {
        particles.reserve(opts.particle_count);
        addParticles(opts.load_mesh);
    }


  void resetGrid() {
    #pragma omp parallel for collapse(3) num_threads(NThreads)
    for (u32 i=0; i < N; i++) {
      for (u32 j=0; j < N; j++) {
        for (u32 k=0; k < N; k++) {
          grid[i][j][k](0) = 0;
          grid[i][j][k](1) = 0;
          grid[i][j][k](2) = 0;
          grid[i][j][k](3) = 0.0;
          //grid[i][j][k] = Vec4::Constant(0.0);
        }
      }
    }
  }

  void advance() {
    resetGrid();
    particleToGridTransfer();
    gridOperations();
    gridToParticleTransfer();
  }

  void particleToGridTransfer() {
    // Particle-to-grid.

    #pragma omp parallel for num_threads(NThreads)
    for (u32 pi = 0; pi < particles.size(); ++pi) {
      Particle & particle = particles[pi];

      TransferScheme transferScheme;
      transferScheme.p2g_prepare_particle(particle,
                                          par,
                                          ParticleVolume,
                                          ParticleMass,
                                          interpolationKernel,
                                          materialModel);

      Veci range_begin = transferScheme.get_range_begin();

      Vec dist_part2node;
      u32 i_begin = std::max(0, -range_begin(0));
      u32 j_begin = std::max(0, -range_begin(1));
      u32 k_begin = std::max(0, -range_begin(2));
      u32 i_end = std::min(interpolationKernel.size(), par.N - range_begin(0));
      u32 j_end = std::min(interpolationKernel.size(), par.N - range_begin(1));
      u32 k_end = std::min(interpolationKernel.size(), par.N - range_begin(2));

      for(u32 i = i_begin; i < i_end; ++i) {
        u32 i_glob = range_begin(0) + i;
        dist_part2node(0) = i_glob * par.dx - particle.x(0);

        for(u32 j = j_begin; j < j_end; ++j) {
          u32 j_glob = range_begin(1) + j;
          dist_part2node(1) = j_glob * par.dx - particle.x(1);

          for(u32 k = k_begin; k < k_end; ++k) {
            u32 k_glob = range_begin(2) + k;
            dist_part2node(2) = k_glob * par.dx - particle.x(2);

            Vec4 node_contribution = transferScheme.p2g_node_contribution(particle, dist_part2node, ParticleMass, i, j, k);

            for(int idx = 0; idx < 4; ++idx) {
              #pragma omp atomic
              grid[i_glob][j_glob][k_glob](idx) += node_contribution(idx);
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

  void gridToParticleTransfer() {
    // Grid-to-particle.
    #pragma omp parallel for num_threads(NThreads)
    for (u32 pi = 0; pi < particles.size(); ++pi) {
      Particle & particle = particles[pi];

      TransferScheme transferScheme;
      transferScheme.g2p_prepare_particle(particle,
                                          par,
                                          interpolationKernel);

      Veci range_begin = transferScheme.get_range_begin();


      Vec dist_part2node;
      u32 i_begin = std::max(0, -range_begin(0));
      u32 j_begin = std::max(0, -range_begin(1));
      u32 k_begin = std::max(0, -range_begin(2));
      u32 i_end = std::min(interpolationKernel.size(), par.N - range_begin(0));
      u32 j_end = std::min(interpolationKernel.size(), par.N - range_begin(1));
      u32 k_end = std::min(interpolationKernel.size(), par.N - range_begin(2));

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
};

int main(int argc, char *argv[]) {
  CLIOptions flags(argc, argv);

  Simulation simulation(flags,
                        MMJelly<MLS_APIC_Particle>(),
                        QuadraticInterpolationKernel(),
                        MLS_APIC_Scheme<QuadraticInterpolationKernel>(),
                        MLS_APIC_Particle());

  ParticleWriter writer;

  Renderer renderer(flags.particle_count, flags.save_dir);

  renderer.render(simulation.particles);

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
  for (unsigned int i = 0; i < 50000; i++) {
    std::cout << "Step " << i << "\r" << std::flush;
    simulation.advance();
    renderer.render(simulation.particles);
    if (save && (i % save_every) == 0) {
      std::stringstream ss;
      ss << flags.save_dir << "/meshes/mesh_" << frame_id << ".obj";
      mesher.computeMesh(ss.str(), simulation.particles);
      ss.str("");
      ss.clear();
      ss << flags.save_dir << "/particles/particles_" << frame_id << ".bgeo";
      std::string filepath = ss.str();
      writer.writeParticles(filepath, simulation.particles);
      frame_id++;
    }
  }
}

