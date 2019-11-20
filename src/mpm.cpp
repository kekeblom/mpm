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
#include <igl/copyleft/marching_cubes.h>
#include <igl/writeOBJ.h>

#include "options.h"
#include "linalg.h"
#include "utils.h"
#include "renderer.h"
#include "types.h"

#include "MaterialModel.h"
#include "InterpolationKernel.h"
#include "TransferScheme.h"

#include "particle_writer.h"

namespace fs = boost::filesystem;

const unsigned int FrameRate = 60;

const real ParticleMass = 1.0;
const real ParticleVolume = 1.0;
const real Gravity = -9.81;
const int NThreads = 8;


float get_random() {
  return float(rand()) / float(RAND_MAX);
}

template<class MaterialModel, class InterpolationKernel, class TransferScheme, class Particle>
class Simulation {
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
      u32 side = int(std::cbrt(particle_count_target));
      real start = opts.N / 3 * par.dx;
      real random_size = opts.N / 3 * par.dx;
      for (u32 i=0; i < side; i++) {
        for (u32 j=0; j < side; j++) {
          for (u32 k=0; k < side; k++) {
            auto x = Vec(start + get_random() * random_size,
                         start + get_random() * random_size,
                         start + get_random() * random_size);
            particles.push_back(Particle(x));
          }
        }
      }
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

  real getWeight(real diff) {
    if (0.0 <= diff && diff < 0.5) {
      return 0.75 - diff * diff;
    } else if (0.5 <= diff && diff < 1.5) {
      auto inner = 1.5 - std::abs(diff);
      return 0.5 * inner * inner;
    } else if (1.5 <= diff) {
      return 0.0;
    } else {
      assert(false);
      return 0.0;
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

          // new velocity:
          // explicit time integration: v^(n+1) = v


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

template <class Particle>
class Mesher {
  private:
    const std::vector<Particle> particles;
    const SimulationParameters& params;
    const u32 VoxelGridSide = 100;
  public:
    Mesher(const std::vector<Particle> &particles, const SimulationParameters& params) : particles(particles), params(params) {}

    void computeMesh(const std::string& filename) {
      boost::multi_array<bool, 3> voxel(boost::extents[VoxelGridSide][VoxelGridSide][VoxelGridSide]);
      double voxel_dx = params.N * params.dx / double(VoxelGridSide);
      for (auto &particle : particles) {
        Vecu32 grid_index = (particle.x / voxel_dx).template cast<u32>();
        if ((grid_index.array() >= VoxelGridSide).any()) {
          continue;
        }
        voxel[grid_index[0]][grid_index[1]][grid_index[2]] = true;
      }
      boost::multi_array<double, 3> sdf(boost::extents[VoxelGridSide][VoxelGridSide][VoxelGridSide]);
      for (u32 i=0; i < VoxelGridSide; i++) {
        for (u32 j=0; j < VoxelGridSide; j++) {
          for (u32 k=0; k < VoxelGridSide; k++) {
            sdf[i][j][k] = double(computeDistance(voxel, i, j, k)) * voxel_dx;
          }
        }
      }
      Eigen::MatrixXd V;
      Eigen::MatrixXi F;
      const int grid_side = int(VoxelGridSide);
      const int point_count = std::pow(grid_side, 3);
      Eigen::VectorXd S(point_count);
      Eigen::MatrixXd GV(point_count, 3);
      for (int i=0; i < grid_side; i++) {
        for (int j=0; j < grid_side; j++) {
          for (int k=0; k < grid_side; k++) {
            const int index = i*grid_side*grid_side + j*grid_side + k;
            S(index, 0) = double(sdf[i][j][k]);
            GV(index, 0) = i;
            GV(index, 1) = j;
            GV(index, 2) = k;
          }
        }
      }
      igl::copyleft::marching_cubes(S, GV, int(VoxelGridSide), int(VoxelGridSide), int(VoxelGridSide), V, F);
      igl::writeOBJ(filename, V, F);
      //for (u32 i=0; i < VoxelGridSide; i++) {
      //  for (u32 j=0; j < VoxelGridSide; j++) {
      //    std::cout << sdf[i][j][u32(VoxelGridSide/2)] << " ";
      //  }
      //  std::cout << std::endl;
      //}
    }

    i32 computeDistance(const boost::multi_array<bool, 3> &voxel, const u32 i, const u32 j, const u32 k) {
      // find closest free cell.
      bool occupied = voxel[i][j][k];
      bool looking_for_value = !occupied;
      i32 grid_length = VoxelGridSide;
      i32 distance = 1;
      do {
        for (i32 delta_i = -distance; delta_i <= distance; delta_i += distance) {
          i32 i_value = i + delta_i;
          if (i_value < 0 || i_value >= grid_length) continue;
          for (i32 delta_j = -distance; delta_j <= distance; delta_j += distance) {
            i32 j_value = j + delta_j;
            if (j_value < 0 || j_value >= grid_length) continue;
            for (i32 delta_k = -distance; delta_k <= distance; delta_k += distance) {
              i32 k_value = k + delta_k;
              if (k_value < 0 || k_value >= grid_length) continue;
              bool value = voxel[i_value][j_value][k_value];
              if (value == looking_for_value) {
                if (occupied) {
                  return distance;
                } else {
                  return -distance;
                }
              }
            }
          }
        }
        distance += 1;
        if (distance >= grid_length) {
          return distance;
        }
      } while (true);
    }
};

int main(int argc, char *argv[]) {
  CLIOptions flags(argc, argv);

  Simulation simulation(flags,
                        MMSnow<MLS_APIC_Particle>(),
                        QuadraticInterpolationKernel(),
                        MLS_APIC_Scheme<QuadraticInterpolationKernel>(),
                        MLS_APIC_Particle());

  ParticleWriter writer;

  Renderer renderer(flags.particle_count, flags.save_dir);

  renderer.render(simulation.particles);

  Mesher<MLS_APIC_Particle> mesher(simulation.particles, simulation.par);

  bool save = flags.save_dir != "";
  if (save) {
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
      mesher.computeMesh(ss.str());
      ss.str("");
      ss.clear();
      ss << flags.save_dir << "/particles/particles_" << frame_id << ".bgeo";
      std::string filepath = ss.str();
      writer.writeParticles(filepath, simulation.particles);
      frame_id++;
    }
  }
}

