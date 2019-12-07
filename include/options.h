#include <cxxopts.h>
#include <types.h>
#ifndef CLI_OPTIONS
#define CLI_OPTIONS

struct CLIOptions {
  // Simulation parameters.
  float dt;
  u32 N;
  u32 particle_count;
  real dx;
  real N_real;
  std::string save_dir;
  std::string load_mesh;

  // Mesh builder parameters.
  u32 mesh_grid;
  u32 mesh_particle_radius;
  i32 mesh_face_count;

  CLIOptions(int argc, char *argv[]) {
    cxxopts::Options parser("Simulator", "MPM simulation.");
    parser.add_options()
      ("dt", "Physics simulation time-step", cxxopts::value<float>()->default_value("1e-4"))
      ("N", "Grid dimensions", cxxopts::value<u32>()->default_value("60"))
      ("save-dir", "Where to save images", cxxopts::value<std::string>()->default_value(""))
      ("particle-count", "Particle density (i.e. number of particles per unit cube)", cxxopts::value<u32>()->default_value("370000"))
      ("load-mesh", "Mesh of object to simulate", cxxopts::value<std::string>()->default_value("../meshes/cube.obj"))
      ("mesh-grid", "Grid size for computing mesh", cxxopts::value<u32>()->default_value("150"))
      ("mesh-particle-radius", "Particle radius for computeing the mesh (in grid points)", cxxopts::value<u32>()->default_value("5"))
      ("mesh-face-count", "Approximate resulting mesh with x faces.", cxxopts::value<i32>()->default_value("-1"));

    try {
      auto flags = parser.parse(argc, argv);
      this->dt = flags["dt"].as<float>();
      this->N = flags["N"].as<u32>();
      this->particle_count = flags["particle-count"].as<u32>();
      this->dx = 1.0 / this->N;
      this->N_real = real(this->N);
      this->save_dir = flags["save-dir"].as<std::string>();
      this->load_mesh = flags["load-mesh"].as<std::string>();
      this->mesh_grid = flags["mesh-grid"].as<u32>();
      this->mesh_particle_radius = flags["mesh-particle-radius"].as<u32>();
      this->mesh_face_count = flags["mesh-face-count"].as<i32>();
    } catch (const cxxopts::OptionException &error) {
      std::cout << error.what() << std::endl;
      exit(1);
    }
  }
};
#endif
