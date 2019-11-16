#include <vector>
#include <iostream>
#include <algorithm>
#include <array>
#include <eigen3/Eigen/Dense>
#include <assert.h>
#include <boost/filesystem.hpp>
#include <boost/multi_array.hpp>
#include <omp.h>
#include <cmath>

#include "options.h"
#include "linalg.h"
#include "utils.h"
#include "renderer.h"

#include "MaterialModel.h"
#include "InterpolationKernel.h"
#include "TransferScheme.h"

#include "particle_writer.h"

namespace fs = boost::filesystem;

const unsigned int FrameRate = 60;

const real particle_mass = 1.0;
const real particle_volume = 1.0;
const real Gravity = -2000.0;




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








template<class MaterialModel, class InterpolationKernel, class TransferScheme, class Particle>
class Simulation {
  private:
	MaterialModel materialModel;
	InterpolationKernel interpolationKernel;
	
    const CLIOptions flags;
	
	SimulationParameters par;
    u32 & N = par.N;
    const u32 particle_count_target;

  public:
    std::vector<Particle> particles;
    boost::multi_array<Vec4, 3> grid; // Velocity x, y, z, mass

    Simulation(const CLIOptions &opts, 
			   MaterialModel const & materialModel, 
			   InterpolationKernel const & interpolationKernel,
			   TransferScheme const & transferScheme_dummy,
			   Particle const & particle_dummy) 
		: flags(opts), 
		  par(opts.dt, opts.N),
//		  N(opts.N), 
		  grid(boost::extents[opts.N][opts.N][opts.N]), 
		  particle_count_target(opts.particle_count),
		  materialModel(materialModel),
		  interpolationKernel(interpolationKernel)
	{
      u32 side = int(std::cbrt(particle_count_target));
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
//            grid[i][j][k](0) = 0;
//            grid[i][j][k](1) = 0;
//            grid[i][j][k](2) = 0;
//            grid[i][j][k](3) = 0.0;
			  grid[i][j][k] = Vec4::Constant(0.0);
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
	
	#pragma omp parallel
	for (u32 pi = 0; pi < particles.size(); ++pi) {
	  Particle & particle = particles[pi];
	  
	  TransferScheme transferScheme;
	  transferScheme.p2g_prepare_particle(particle, 
										  par, 
										  particle_volume,
										  particle_mass,
										  interpolationKernel,
										  materialModel);
	  
	  Veci range_begin = transferScheme.get_range_begin();
	  
	  
	  Vec dist_part2node;
	  u32 i_begin = std::max(0, -range_begin(0));
	  u32 j_begin = std::max(0, -range_begin(1));
	  u32 k_begin = std::max(0, -range_begin(2));
	  u32 i_end = std::min(interpolationKernel.size(), flags.N - range_begin(0));
	  u32 j_end = std::min(interpolationKernel.size(), flags.N - range_begin(1));
	  u32 k_end = std::min(interpolationKernel.size(), flags.N - range_begin(2));
	  
	  for(u32 i = i_begin; i < i_end; ++i) {
		  u32 i_glob = range_begin(0) + i;
		  dist_part2node(0) = i_glob * flags.dx - particle.x(0);
		  
		  for(u32 j = j_begin; j < j_end; ++j) {
			  u32 j_glob = range_begin(1) + j;
			  dist_part2node(1) = j_glob * flags.dx - particle.x(1);
			  
			  for(u32 k = k_begin; k < k_end; ++k) {
				  u32 k_glob = range_begin(2) + k;
				  dist_part2node(2) = k_glob * flags.dx - particle.x(2);
				  
				  Vec4 node_contribution = transferScheme.p2g_node_contribution(particle, dist_part2node, particle_mass, i, j, k);
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
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        for (size_t k = 0; k < N; k++) {
          Vec4 &cell = grid[i][j][k];
		  
		  // boundary collisions
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
		  
		  // new velocity:
		  // explicit time integration: v^(n+1) = v
		  

        }
      }
    }
  }

  void gridToParticleTransfer() {
    // Grid-to-particle.
    #pragma omp parallel for
    for (u32 pi = 0; pi < particles.size(); ++pi) {
      Particle & particle = particles[pi];
		
      Eigen::Matrix<u32, 3, 1> base_coordinate = (static_cast<ParticleBase>(particle).x * par.N_real - Vec::Ones() * 0.5).cast<u32>();

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
			
//			if(std::isnan(particle.v(0)) || std::isnan(particle.v(1)) || std::isnan(particle.v(2))) {
//				assert(false);
//			}
			
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
	  
	  if(std::abs(particle.F.determinant()) < 1.0e-9) {
		  assert(false);
	  }
	  
	  
    }
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

