#include <vector>
#include <iostream>
#include <algorithm>
#include <array>
#include <eigen3/Eigen/Dense>
#include <assert.h>
#include <boost/filesystem.hpp>
#include <boost/multi_array.hpp>
#include <omp.h>
#include <Partio.h>
#include "options.h"
#include "linalg.h"
#include "utils.h"
#include "renderer.h"
#include "particle_writer.h"

namespace fs = boost::filesystem;

const unsigned int FrameRate = 60;

const real particle_mass = 1.0;
const real particle_volume = 1.0;
const real hardening = 10.0;
const real E = 1e5;
const real Nu = 0.3;
const real Gravity = -2000.0;

const real Mu0 = E / (2 * (1 + Nu));
const real Lambda0 = E * Nu / ((1 + Nu) * (1 - 2 * Nu));

using Vec4 = Eigen::Matrix<real, 4, 1>;

real clamp(const real &number, const real &lower, const real &upper) {
  return std::max(std::min(number, upper), lower);
}

float get_random() {
  return float(rand()) / float(RAND_MAX);
}

template <class T>
T square(const T& value) {
  return value * value;
}

class Simulation {
  private:
    const CLIOptions flags;
    const u32 N;
    const u32 particle_count;

  public:
    std::vector<Particle> particles;
    boost::multi_array<Vec4, 3> grid; // Velocity x, y, z, mass

    Simulation(const CLIOptions &opts) : flags(opts), N(opts.N), particle_count(opts.particle_count), grid(boost::extents[opts.N][opts.N][opts.N]) {
      u32 side = int(std::cbrt(particle_count));
      real start = opts.N / 3 * flags.dx;
      real random_size = opts.N / 3 * flags.dx;
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
      #pragma omp parallel for
      for (u32 i=0; i < N; i++) {
        for (u32 j=0; j < N; j++) {
          for (u32 k=0; k < N; k++) {
            grid[i][j][k](0) = 0;
            grid[i][j][k](1) = 0;
            grid[i][j][k](2) = 0;
            grid[i][j][k](3) = 1.0;
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
    for (u32 pi=0; pi < particle_count; pi++) {
      Particle &particle = particles[pi];
      Eigen::Matrix<u32, 3, 1> base_coordinate = (particle.x * flags.N_real - Vec::Ones() * 0.5).cast<u32>();

      assert(base_coordinate(1) < N);

      Vec fx = particle.x * flags.N_real - base_coordinate.cast<real>();

      real e = std::exp(hardening * (1.0 - particle.Jp));
      real mu = Mu0 * e;
      real lambda = Lambda0 * e;

      real J = particle.Jp;

      Mat R, S;
      polar_decomposition(particle.F, R, S);

      real Dinv = 4 * flags.N_real * flags.N_real;
      Mat PF = (2 * mu * (particle.F - R) * particle.F.transpose()) + lambda * ((J - 1) * J) * (particle.F.inverse().transpose());

      Mat stress = -Dinv * flags.dt * particle_volume * PF * particle.F.transpose();

      Mat affine = stress + particle_mass * particle.C;

      //auto w = computeWeights(fx);

      auto grid_x = particle.x * flags.N_real;

      u32 until_i = std::min<u32>(base_coordinate(0) + 3, N-1);
      u32 until_j = std::min<u32>(base_coordinate(1) + 3, N-1);
      u32 until_k = std::min<u32>(base_coordinate(2) + 3, N-1);
      for (u32 i = base_coordinate(0); i < until_i; i++) {
        real weight_i = getWeight(std::abs(i - grid_x(0)));
        u32 relative_i = i - base_coordinate(0);
        for (u32 j = base_coordinate(1); j < until_j; j++) {
          u32 relative_j = j - base_coordinate(1);
          real weight_j = getWeight(std::abs(j - grid_x(1)));
          for (u32 k = base_coordinate(2); k < until_k; k++) {
            real weight_k = getWeight(std::abs(k - grid_x(2)));
            u32 relative_k = k - base_coordinate(2);
            Vec position_difference = (Vec(relative_i, relative_j, relative_k) - fx) * flags.dx;
            Vec momentum = particle.v * particle_mass;
            auto &cell = grid[i][j][k];
            Vec momentum_affine_delta_pos = momentum + affine * position_difference;
            Vec4 momentum_mass(momentum_affine_delta_pos(0),
                momentum_affine_delta_pos(1),
                momentum_affine_delta_pos(2),
                particle_mass);
            real weight = weight_i * weight_j * weight_k;
            //real weight = w[relative_i][0] * w[relative_j][1] * w[relative_k][2];
            cell += (
              weight * momentum_mass
            );
          }
        }
      }
    }
  }

  void gridOperations() {
    // Grid operations.
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        for (size_t k = 0; k < N; k++) {
          Vec4 &cell = grid[i][j][k];

          if (cell[3] > 0.0) {
            cell /= cell[3];

            cell += flags.dt * Vec4(0, Gravity, 0, 0);

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
    #pragma omp parallel for
    for (u32 pi=0; pi < particle_count; pi++) {
      Particle &particle = particles[pi];
      Eigen::Matrix<u32, 3, 1> base_coordinate = (particle.x * flags.N_real - Vec::Ones() * 0.5).cast<u32>();

      Vec fx = particle.x * flags.N_real - base_coordinate.cast<real>();

      particle.C = Mat::Zero();
      particle.v = Vec::Zero();

      auto grid_x = particle.x * flags.N_real;

      u32 until_i = std::min<u32>(base_coordinate(0) + 3, N-1);
      u32 until_j = std::min<u32>(base_coordinate(1) + 3, N-1);
      u32 until_k = std::min<u32>(base_coordinate(2) + 3, N-1);
      for (u32 i = base_coordinate(0); i < until_i; i++) {
        u32 relative_i = i - base_coordinate(0);
        real weight_i = getWeight(std::abs(i - grid_x(0)));
        for (u32 j = base_coordinate(1); j < until_j; j++) {
          u32 relative_j = j - base_coordinate(1);
          real weight_j = getWeight(std::abs(j - grid_x(1)));
          for (u32 k = base_coordinate(2); k < until_k; k++) {
            real weight_k = getWeight(std::abs(k - grid_x(2)));
            u32 relative_k = k - base_coordinate(2);
            Vec position_diff = (Vec(relative_i, relative_j, relative_k) - fx) * flags.dx;
            auto &grid_cell = grid[i][j][k];
            Vec grid_velocity = Vec(grid_cell[0], grid_cell[1], grid_cell[2]);
            real weight = weight_i * weight_j * weight_k;
            Vec weighted_grid_velocity = weight * grid_velocity;
            particle.v += weighted_grid_velocity;
            Eigen::Matrix<real, 3, 3> C_diff = 4 * flags.N_real * (weighted_grid_velocity * position_diff.transpose());
            particle.C += C_diff;
          }
        }
      }
      particle.x += flags.dt * particle.v;

      Mat F = (Mat::Identity() + flags.dt * particle.C) * particle.F;
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
      particle.F = F;
    }
  }
};

int main(int argc, char *argv[]) {
  CLIOptions flags(argc, argv);
  Simulation simulation(flags);
  ParticleWriter writer;

  Renderer renderer(flags.particle_count, flags.save_dir);

  renderer.render(simulation.particles);

  bool save = flags.save_dir != "";
  if (save) {
    fs::create_directory(flags.save_dir);
  }

  for (unsigned int i = 0; i < 50000; i++) {
    std::cout << "Step " << i << "\r" << std::flush;
    simulation.advance();
    renderer.render(simulation.particles);
    if (save) {
      std::stringstream ss;
      ss << flags.save_dir << "/particles_" << i << ".bgeo";
      std::string filepath = ss.str();
      writer.writeParticles(filepath, simulation.particles);
    }
  }
}

