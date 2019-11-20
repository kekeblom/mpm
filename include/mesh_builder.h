#include <boost/multi_array.hpp>
#include <igl/copyleft/marching_cubes.h>
#include <igl/writeOBJ.h>
#include <chrono>
#include <omp.h>

#include "types.h"
#include "TransferScheme.h"

template <class Particle>
class MeshBuilder {
  private:
    const std::vector<Particle> particles;
    const SimulationParameters& params;
    const u32 VoxelGridSide;
  public:
    MeshBuilder(const std::vector<Particle>&, const SimulationParameters&, const u32);

    void computeMesh(const std::string&);

  private:
    i32 computeDistance(const boost::multi_array<bool, 3> &, const u32, const u32, const u32);
};


template <class Particle>
MeshBuilder<Particle>::MeshBuilder(const std::vector<Particle> &particles, const SimulationParameters& params, const u32 grid_size)
  : particles(particles), params(params), VoxelGridSide(grid_size) {};

template<class Particle>
void MeshBuilder<Particle>::computeMesh(const std::string& filename) {
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

  #pragma omp parallel for collapse(3) schedule(dynamic, 25)
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
}

template<class Particle>
i32 MeshBuilder<Particle>::computeDistance(const boost::multi_array<bool, 3> &voxel, const u32 i, const u32 j, const u32 k) {
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
      return -grid_length;
    }
  } while (true);
}

