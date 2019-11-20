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
    i32 computeDistance(const boost::multi_array<bool, 3> &, const i32, const i32, const i32);
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
  const int grid_side = int(VoxelGridSide);

  #pragma omp parallel for collapse(3) schedule(dynamic, 25)
  for (i32 i=0; i < grid_side; i++) {
    for (i32 j=0; j < grid_side; j++) {
      for (i32 k=0; k < grid_side; k++) {
        sdf[i][j][k] = double(computeDistance(voxel, i, j, k)) * voxel_dx;
      }
    }
  }
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
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
  igl::copyleft::marching_cubes(S, GV, grid_side, grid_side, grid_side, V, F);
  igl::writeOBJ(filename, V, F);
}

template<class Particle>
i32 MeshBuilder<Particle>::computeDistance(const boost::multi_array<bool, 3> &voxel, const i32 i, const i32 j, const i32 k) {
  // find closest free cell.
  bool occupied = voxel[i][j][k];
  bool looking_for_value = !occupied;
  i32 grid_length = VoxelGridSide;
  i32 distance = 1;
  bool v[6] = {false, false, false, false, false, false};
  do {
    if (i-distance >= 0) {
      v[0] = voxel[i-distance][j][k];
    }
    if (i+distance < grid_length) {
      v[1] = voxel[i+distance][j][k];
    }
    if (j-distance >= 0) {
      v[2] = voxel[i][j-distance][k];
    }
    if (j+distance < grid_length) {
      v[3] = voxel[i][j+distance][k];
    }
    if (k-distance >= 0) {
      v[4] = voxel[i][j][k-distance];
    }
    if (k+distance < grid_length) {
      v[5] = voxel[i][j][k+distance];
    }

    if (v[0] == looking_for_value ||
        v[1] == looking_for_value ||
        v[2] == looking_for_value ||
        v[3] == looking_for_value ||
        v[4] == looking_for_value ||
        v[5] == looking_for_value) {
      if (looking_for_value) {
        return -distance;
      } else {
        return distance;
      }
    }
    distance += 1;
    if (distance >= grid_length) {
      return -grid_length;
    }
  } while (true);
}

