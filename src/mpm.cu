#include "mpm.cuh"
#include "gpu.h"

namespace fs = boost::filesystem;

const real Gravity = -9.81;
const int GridVectorSize = 4;
const int ParticleBlockSize = 64;

float get_random() {
  return float(rand()) / float(RAND_MAX);
}

__global__ void zeroGrid(Vec4* grid) {
  int N = gridDim.x;
  int max_index = N * N * N;
  int base_index = N * N * blockIdx.x + N * threadIdx.x;
  for (int i=0; i < N; i++) {
    int index = base_index +  i;
    if (index >= max_index) return;
    Vec4& cell = grid[index];
    for (int j=0; j < GridVectorSize; j++) {
      cell[j] = 0.0;
    }
  }
}

__global__ void particleToGrid(Particle* particles, Vec4* grid, MaterialModel* material_models,
    int cutoff, SimulationParameters *parameters, InterpolationKernel *interpolation_kernel) {
  __shared__ Particle block_particles[ParticleBlockSize];
  int pi = blockIdx.x * blockDim.x + threadIdx.x;
  float dx = parameters->dx;
  int N = parameters->N;
  if (pi >= cutoff) return;
  block_particles[threadIdx.x] = particles[pi];
  Particle & particle = block_particles[threadIdx.x];
  MaterialModel material_model = material_models[particle.material_type];

  TransferScheme transferScheme;
  transferScheme.p2g_prepare_particle(particle, *parameters, *interpolation_kernel, material_model);

  Veci range_begin = transferScheme.get_range_begin();  // get start of range in grid that is influenced by the particle

  // handle particles that are completely outside of the domain
  for (int i = 0; i < 3; ++i) {
    if (range_begin(i) + int(interpolation_kernel->size()) < 0 || range_begin(i) >= int(N)) {
      return;
    }
  }

  // bounds of particle-influence
  Vec particle_node_distance;
  u32 i_begin = max(0, -range_begin(0));
  u32 j_begin = max(0, -range_begin(1));
  u32 k_begin = max(0, -range_begin(2));
  u32 i_end = min(interpolation_kernel->size(), N + range_begin(0));
  u32 j_end = min(interpolation_kernel->size(), N + range_begin(1));
  u32 k_end = min(interpolation_kernel->size(), N + range_begin(2));

  // loop through relevant grid cells
  for (u32 i = i_begin; i < i_end; ++i) {
    u32 i_glob = range_begin[0] + i;
    u32 index_i = N * N * i_glob;

    particle_node_distance[0] = i_glob * dx - particle.x[0];

    for (u32 j = j_begin; j < j_end; ++j) {
      u32 j_glob = range_begin[1] + j;
      u32 index_j = N * j_glob;
      particle_node_distance[1] = j_glob * dx - particle.x[1];

      for (u32 k = k_begin; k < k_end; ++k) {
        u32 k_glob = range_begin[2] + k;
        particle_node_distance[2] = k_glob * dx - particle.x[2];

        Vec4 node_contribution = transferScheme.p2g_node_contribution(particle, particle_node_distance, material_model.particleMass, i, j, k);

        int index = index_i + index_j + k_glob;
        Vec4& cell = grid[index]; //getCell(i_glob, j_glob, k_glob);
        for (int idx = 0; idx < GridVectorSize; idx++) {
          atomicAdd(&(cell[idx]), node_contribution[idx]);
        }
      }
    }
  }
}

__global__ void gridOpKernel(Vec4* grid, int cutoff, float dt) {
  int N = blockDim.x;
  int x_index = blockIdx.x;
  int y_index = blockIdx.y;
  int z_index = threadIdx.x;
  int index = N * (N * x_index + y_index) + z_index;
  if (index >= N*N*N) return;
  Vec4& cell = grid[index];
  // boundary collisions
  if (cell[3] > 0.0) {
    for (int w=0; w < GridVectorSize; w++) {
      cell[w] /= cell[3];
    }

    cell[1] += dt * Gravity;

    const real boundary = 0.05;

    const real x = real(x_index) / N;
    const real y = real(y_index) / N;
    const real z = real(z_index) / N;
    if (x < boundary || x > 1-boundary || y > 1-boundary || z < boundary || z > 1-boundary) {
      cell[0] = 0.0;
      cell[1] = 0.0;
      cell[2] = 0.0;
      cell[3] = cell[3];
    }
    if (y < boundary) {
      cell[1] = max(real(0.0), cell[1]);
    }
  }
}

__global__ void gridToParticle(Vec4* grid, Particle* particles, MaterialModel* material_models, SimulationParameters &parameters,
    InterpolationKernel &interpolation_kernel) {
  __shared__ Particle block_particles[ParticleBlockSize];
  int pi = blockIdx.x * blockDim.x + threadIdx.x;
  int N = parameters.N;
  if (pi > N * N * N) return;

  block_particles[threadIdx.x] = particles[pi];

  Particle& particle = block_particles[threadIdx.x];
  MaterialModel& material_model = material_models[particle.material_type];

  TransferScheme transferScheme;
  transferScheme.g2p_prepare_particle(particle,
                                      parameters,
                                      interpolation_kernel);

  Veci range_begin = transferScheme.get_range_begin(); // get start of range in grid that is influenced by the particle

  // handle particles that are completely outside of the domain
  for(int i = 0; i < 3; ++i) {
    if(range_begin(i)+int(interpolation_kernel.size()) < 0 || range_begin(i) >= int(N)) {
      return;
    }
  }

  // bounds of particle-influence
  Vec dist_part2node;
  u32 i_begin = max(0, -range_begin(0));
  u32 j_begin = max(0, -range_begin(1));
  u32 k_begin = max(0, -range_begin(2));
  u32 i_end = min(interpolation_kernel.size(), N - range_begin(0));
  u32 j_end = min(interpolation_kernel.size(), N - range_begin(1));
  u32 k_end = min(interpolation_kernel.size(), N - range_begin(2));

  // loop through relevant grid cells
  for(u32 i = i_begin; i < i_end; ++i) {
    u32 i_glob = range_begin(0) + i;
    dist_part2node[0] = i_glob * parameters.dx - particle.x(0);
    u32 index_i = N * N * i_glob;
    for(u32 j = j_begin; j < j_end; ++j) {
      u32 j_glob = range_begin(1) + j;
      dist_part2node[1] = j_glob * parameters.dx - particle.x(1);
      u32 index_j = N * j_glob;

      for(u32 k = k_begin; k < k_end; ++k) {
        u32 k_glob = range_begin(2) + k;
        dist_part2node[2] = k_glob * parameters.dx - particle.x(2);

        // actual transfer
        // velocity
        Vec4& cell = grid[index_i + index_j + k_glob];
        transferScheme.g2p_node_contribution(particle,
                                             dist_part2node,
                                             cell,
                                             i, j, k);
      }
    }
  }

  transferScheme.g2p_finish_particle(particle, parameters);

  // plasticity
  material_model.endOfStepMutation(particle);

  // advection
  particle.x += parameters.dt * particle.v;

  particles[pi] = particle;
}

Simulation::Simulation(const CLIOptions &opts,
           InterpolationKernel const & interpolationKernel, std::vector<MaterialModel> const &material_models) :
    par(opts.dt, opts.N),
    interpolationKernel(interpolationKernel),
    material_models(material_models) {
  N2 = par.N * par.N;
  sizeof_grid = sizeof(Vec4) * N * N * N;
  device_particles = nullptr;
}

Simulation::~Simulation() {
  cudaFree(device_grid);
  cudaFree(device_particles);
  cudaFree(device_material_models);
  delete[] host_particles;
}

void Simulation::initCuda() {
  int bytes = material_models.size() * sizeof(MaterialModel);
  cudaMalloc((void **)&device_grid, par.N * par.N * par.N * sizeof(Vec4));
  cudaMalloc((void **)&device_material_models, bytes);
  cudaMemcpy(device_material_models, material_models.data(), bytes, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&device_interpolation_kernel, sizeof(InterpolationKernel));
  cudaMemcpy(device_interpolation_kernel, &interpolationKernel, sizeof(InterpolationKernel), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&device_parameters, sizeof(SimulationParameters));
  cudaMemcpy(device_parameters, &par, sizeof(SimulationParameters), cudaMemcpyHostToDevice);
  particlesToDevice();
}

void Simulation::syncDevice() {
  particlesToHost();
}

void Simulation::resetGrid() {
  zeroGrid<<<int(N), int(N)>>>(device_grid);
}

void Simulation::particleToGridTransfer(std::vector<SimObject> & objects) {
  int n_threads = ParticleBlockSize;
  int grid_width = std::ceil(float(active_particle_count) / float(n_threads));
  particleToGrid<<<grid_width, n_threads>>>(device_particles, device_grid, device_material_models,
      active_particle_count, device_parameters, device_interpolation_kernel);
}

void Simulation::gridOperations() {
  dim3 blocks(N, N);
  int threads = N;
  gridOpKernel<<<blocks, threads>>>(device_grid, N*N*N, par.dt);
}

void Simulation::gridToParticleTransfer(std::vector<SimObject> & objects) {
  int n_threads = ParticleBlockSize;
  int grid_width = std::ceil(float(active_particle_count) / float(n_threads));
  gridToParticle<<<grid_width, n_threads>>>(device_grid, device_particles, device_material_models,
      *device_parameters, *device_interpolation_kernel);
}

void Simulation::addObject(std::string const & filepath,  // obj file defining the shape of the object
               int material_model_index,
               real size,                     // size of the object, measured as the longest edge of the bounding box
               Vec position,                  // position of the object within the scene (lowest corner of object)
               Vec velocity,                  // initial velocity of the object
               real lifetime_begin,
               real lifetime_end) {
  auto mesh = loadMesh(filepath, size, position);
  auto material = material_models[material_model_index];
  auto object = SimObject(material);
  objects.push_back(SimObject(material));
  addParticles(mesh, u32(1.0 / material.particleVolume), velocity, objects.back().particles, material_model_index);
  objects.back().lifetime_begin = lifetime_begin;
  objects.back().lifetime_end = lifetime_end;
}

size_t Simulation::getFullParticleCount() {
  size_t n = 0;
  for(auto & object : objects) {
    n += object.particles.size();
  }
  return n;
}

std::vector<Particle> & Simulation::getFullParticleList() {
  particles_all.resize(0);
  for(auto & object : objects) {
    particles_all.insert(particles_all.end(), object.particles.begin(), object.particles.end());
  }
  return particles_all;
}

std::vector<Particle> & Simulation::getActiveParticleList() {
  particles_all.resize(0);
  for (auto & object : objects) {
    if (!object.isActive(t)) continue;
    particles_all.insert(particles_all.end(), object.particles.begin(), object.particles.end());
  }
  return particles_all;
}

void Simulation::particlesToDevice() {
  std::vector<Particle> &particles = getActiveParticleList();
  if (particles.size() != active_particle_count) {
    active_particle_count = particles.size();
    host_particles = new Particle[active_particle_count];
    cudaMalloc((void **)&device_particles, active_particle_count * sizeof(Particle));
  }
  cudaMemcpy(device_particles, particles.data(), active_particle_count * sizeof(Particle), cudaMemcpyHostToDevice);
}

void Simulation::particlesToHost() {
  cudaMemcpy(host_particles, device_particles, active_particle_count * sizeof(Particle), cudaMemcpyDeviceToHost);
  int i = 0;
  for (SimObject& object : objects) {
    if (!object.isActive(t)) continue;
    for (int j=0; j < object.particles.size(); j++) {
      object.particles[j] = host_particles[i];
      i++;
      if (i > active_particle_count) {
        active_particle_count = i;
        reallocateParticles();
        return;
      }
    }
  }
  if (i != active_particle_count) {
    std::cout << "Amount of active particles changed" << std::endl;
  }
}

void Simulation::reallocateParticles() {
  cudaFree(device_particles);
  delete[] host_particles;
  host_particles = new Particle[active_particle_count];
  cudaMalloc((void **)&device_particles, active_particle_count * sizeof(Particle));
  particlesToDevice();
}

void printEnd(const char* name, std::chrono::time_point<std::chrono::high_resolution_clock>& start) {
  auto now = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = now - start;
  std::cout << name << " took " << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() / 10000.0 << "s" << std::endl;
}

#include <chrono>
void Simulation::advance() {
  resetGrid();
  particleToGridTransfer(objects);
  gridOperations();
  gridToParticleTransfer(objects);
  t += par.dt;
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXi> Simulation::loadMesh(const std::string& filepath, double size, Vec position) {
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

void Simulation::addParticles(std::pair<Eigen::MatrixXf, Eigen::MatrixXi> const & mesh,
                  u32 particle_density,
                  Vec velocity,
                  std::vector<Particle> & particles,
                  u8 material_index) {
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
        particles.push_back(Particle(material_index, x, velocity));
      }
      if (count_tot == particle_count_target) {
        return;
      }
    }
  }
}

