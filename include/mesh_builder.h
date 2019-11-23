#include <boost/multi_array.hpp>
#include <igl/copyleft/marching_cubes.h>
#include <igl/writeOBJ.h>
#include <igl/decimate.h>
#include <chrono>
#include <omp.h>

#include "types.h"
#include "TransferScheme.h"
#include "InterpolationKernel.h"

template <class Particle>
class MeshBuilder {
  private:
    const std::vector<Particle> particles;
    const SimulationParameters& params;
    const CLIOptions flags;
    const u32 VoxelGridSide;
  public:
    MeshBuilder(const std::vector<Particle>&, const SimulationParameters&, const CLIOptions&, const u32);

    void computeMesh(const std::string&);
};



template <class InterpolationKernel, class Particle>
void fillVoxelGrid_weights(std::vector<Particle> const & particles, double dx, boost::multi_array<float, 3> & grid)
{
  // comutes the sum of weights the particles contribute to the grid
  
  InterpolationKernel interpolationKernel;
  u32 N1 =grid.shape()[0];
  u32 N2 =grid.shape()[1];
  u32 N3 =grid.shape()[2];
  
  // set to zero
  for (u32 i=0; i < N1; i++) {
    for (u32 j=0; j < N2; j++) {
      for (u32 k=0; k < N3; k++) {
        grid[i][j][k] = 0;
      }
    }
  }
  
  for (u32 pi = 0; pi < particles.size(); ++pi) {
    Particle const & particle = particles[pi];
    
    Eigen::Matrix<real, 3, InterpolationKernel::size()> weights;
    Veci range_begin;
    weights = interpolationKernel.weights_per_direction(particle.x, 1.0/dx, range_begin);
    
    u32 i_begin = std::max(0, -range_begin(0));
    u32 j_begin = std::max(0, -range_begin(1));
    u32 k_begin = std::max(0, -range_begin(2));
    u32 i_end = std::min(interpolationKernel.size(), N1 - range_begin(0));
    u32 j_end = std::min(interpolationKernel.size(), N2 - range_begin(1));
    u32 k_end = std::min(interpolationKernel.size(), N3 - range_begin(2));
    
    for(u32 i = i_begin; i < i_end; ++i) {
      u32 i_glob = range_begin(0) + i;
      for(u32 j = j_begin; j < j_end; ++j) {
        u32 j_glob = range_begin(1) + j;
        for(u32 k = k_begin; k < k_end; ++k) {
          u32 k_glob = range_begin(2) + k;
          double weight = weights(0, i) * weights(1, j) * weights(2, k);
            #pragma omp atomic
            grid[i_glob][j_glob][k_glob] += weight;
        }
      }
    }
    
  }
  
}

template <class Particle>
void fillVoxelGrid_distance(std::vector<Particle> const & particles, real dx, real distance_cutoff, boost::multi_array<float, 3> & grid)
{
  u32 N1 =grid.shape()[0];
  u32 N2 =grid.shape()[1];
  u32 N3 =grid.shape()[2];
  
  // set to cutoff distance
  for (u32 i=0; i < N1; i++) {
    for (u32 j=0; j < N2; j++) {
      for (u32 k=0; k < N3; k++) {
        grid[i][j][k] = distance_cutoff;
      }
    }
  }
  
  for (u32 pi = 0; pi < particles.size(); ++pi) {
    Particle const & particle = particles[pi];
    
    real r_gridpoints = distance_cutoff / dx;
    Vec x_particle_gridpoints = particle.x / dx;
    
    // Note: range_begin / end in absolute numbers, with respect to the grid
    Veci range_begin = (x_particle_gridpoints - Vec::Constant(r_gridpoints)).cast<int>();
    Veci range_end = (x_particle_gridpoints + Vec::Constant(r_gridpoints + 1.0)).cast<int>();
    u32 i_begin = std::max(0, range_begin(0));
    u32 j_begin = std::max(0, range_begin(1));
    u32 k_begin = std::max(0, range_begin(2));
    u32 i_end = std::min(int(N1), range_end(0));
    u32 j_end = std::min(int(N2), range_end(1));
    u32 k_end = std::min(int(N3), range_end(2));
    
    Vec x_node;
    for(u32 i = i_begin; i < i_end; ++i) {
      x_node(0) = i * dx;
      for(u32 j = j_begin; j < j_end; ++j) {
        x_node(1) = j * dx;
        for(u32 k = k_begin; k < k_end; ++k) {
          x_node(2) = k * dx;
            real d = (particle.x - x_node).norm();
            grid[i][j][k] = std::min(d, grid[i][j][k]);
        }
      }
    }
    
  }
  
}


template <class Particle>
void fillVoxelGrid_binary(std::vector<Particle> const & particles, real voxel_dx, u32 VoxelGridSide, boost::multi_array<float, 3> & sdf)
{
    for (u32 i=0; i < VoxelGridSide; i++) {
      for (u32 j=0; j < VoxelGridSide; j++) {
        for (u32 k=0; k < VoxelGridSide; k++) {
          sdf[i][j][k] = -1.0f;
        }
      }
    }
    for (auto &particle : particles) {
      Vecu32 grid_index = (particle.x / voxel_dx).template cast<u32>();
      if ((grid_index.array() >= VoxelGridSide).any()) {
        continue;
      }
      sdf[grid_index[0]][grid_index[1]][grid_index[2]] = 1.0f;
    }
}



template <class Particle>
MeshBuilder<Particle>::MeshBuilder(const std::vector<Particle> &particles, const SimulationParameters& params, const CLIOptions& flags, const u32 grid_size)
  : particles(particles), params(params), flags(flags), VoxelGridSide(grid_size) {}

template<class Particle>
void MeshBuilder<Particle>::computeMesh(const std::string& filename) {
  double voxel_dx = params.N * params.dx / double(VoxelGridSide);
  boost::multi_array<float, 3> sdf(boost::extents[VoxelGridSide][VoxelGridSide][VoxelGridSide]);

  const int grid_side = int(VoxelGridSide);
  
  //fillVoxelGrid_binary(particles, voxel_dx, VoxelGridSide, sdf);
  //fillVoxelGrid_weights<QuadraticInterpolationKernel>(particles, voxel_dx, sdf);
  fillVoxelGrid_distance(particles, voxel_dx, (flags.mesh_particle_radius+1)*voxel_dx, sdf);
  
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  const int point_count = std::pow(grid_side, 3);
  Eigen::VectorXd S(point_count);
  Eigen::MatrixXd GV(point_count, 3);
  for (int i=0; i < grid_side; i++) {
    for (int j=0; j < grid_side; j++) {
      for (int k=0; k < grid_side; k++) {
        const int index = i*grid_side*grid_side + j*grid_side + k;
        S(index, 0) = flags.mesh_particle_radius*voxel_dx - double(sdf[i][j][k]);
        GV(index, 0) = i;
        GV(index, 1) = j;
        GV(index, 2) = k;
      }
    }
  }
  igl::copyleft::marching_cubes(S, GV, grid_side, grid_side, grid_side, V, F);
  V = V * voxel_dx;
  
  if (flags.mesh_face_count != -1) {
    Eigen::MatrixXd U;
    Eigen::MatrixXi G;
    Eigen::VectorXi J;
    igl::decimate(V, F, flags.mesh_face_count, U, G, J);
    igl::writeOBJ(filename, U, G);
  } else {
    igl::writeOBJ(filename, V, F);
  }
}

