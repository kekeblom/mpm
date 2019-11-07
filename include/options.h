#include <cxxopts.h>
#include <types.h>
#ifndef CLI_OPTIONS
#define CLI_OPTIONS

struct CLIOptions {
  float dt;
  u32 N;
  u32 particle_count;
  real dx;
  real N_real;
  std::string save_dir;

  CLIOptions(int argc, char *argv[]) {
    cxxopts::Options parser("Simulator", "MPM simulation.");
    parser.add_options()
      ("dt", "Physics simulation time-step", cxxopts::value<float>()->default_value("1e-4"))
      ("N", "Grid dimensions", cxxopts::value<u32>()->default_value("60"))
      ("save-dir", "Where to save images", cxxopts::value<std::string>()->default_value(""))
      ("particle-count", "How many particles to simulate", cxxopts::value<u32>()->default_value("10000"));
    try {
      auto flags = parser.parse(argc, argv);
      this->dt = flags["dt"].as<float>();
      this->N = flags["N"].as<u32>();
      this->particle_count = flags["particle-count"].as<u32>();
      this->dx = 1.0 / this->N;
      this->N_real = real(this->N);
      this->save_dir = flags["save-dir"].as<std::string>();
    } catch (const cxxopts::OptionException &error) {
      std::cout << error.what() << std::endl;
      exit(1);
    }
  }
};
#endif
