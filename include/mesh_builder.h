#include <boost/multi_array.hpp>
#include <igl/copyleft/marching_cubes.h>
#include <igl/writeOBJ.h>
#include <igl/decimate.h>
#include <chrono>
#include <omp.h>

#include "types.h"
#include "TransferScheme.h"

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


template <class Particle>
MeshBuilder<Particle>::MeshBuilder(const std::vector<Particle> &particles, const SimulationParameters& params, const CLIOptions& flags, const u32 grid_size)
  : particles(particles), params(params), flags(flags), VoxelGridSide(grid_size) {};

template<class Particle>
void MeshBuilder<Particle>::computeMesh(const std::string& filename) {
  double voxel_dx = params.N * params.dx / double(VoxelGridSide);
  boost::multi_array<float, 3> sdf(boost::extents[VoxelGridSide][VoxelGridSide][VoxelGridSide]);
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
  const int grid_side = int(VoxelGridSide);

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
  V = V * voxel_dx;
  if (flags.mesh_face_count != -1) {
    Eigen::MatrixXd U;
    Eigen::MatrixXi G;
    Eigen::VectorXi J;
    igl::decimate(V, F, 1000, U, G, J);
    igl::writeOBJ(filename, U, G);
  } else {
    igl::writeOBJ(filename, V, F);
  }
}

