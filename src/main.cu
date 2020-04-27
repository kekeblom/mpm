#include <cpptoml.h>
#include <boost/filesystem.hpp>
#include "types.h"
#include "mpm.cuh"
#include "mesh_builder.h"
#include "particle_writer.h"

const unsigned int FrameRate = 240; // for export of data
std::vector<MaterialModel> material_models;
namespace fs = boost::filesystem;

Vec toVector(cpptoml::option<std::vector<double>> &array) {
  Vec out;
  int i = 0;
  for (const double value : *array) {
    out[i] = value;
    i++;
  }
  return out;
}

int main(int argc, char *argv[]) {
  CLIOptions flags(argc, argv);

  Simulation simulation(flags, InterpolationKernel(), material_models);

  fs::path scene_path(flags.scene);
  std::string meshes_dir = fs::system_complete(scene_path.parent_path() / "meshes/").string();

  std::cout << "Loading scene file: " << flags.scene << std::endl;
  auto config = cpptoml::parse_file(flags.scene);
  auto materials = config->get_table_array("material");
  std::map<std::string, u8> material_index;
  int i = 0;
  for (const auto &material : *materials) {
    MaterialModel model(1.0 / flags.particle_count,
                  material->get_as<double>("density").value_or(700.0),
                  material->get_as<double>("E").value_or(1.4e5),
                  material->get_as<double>("Nu").value_or(0.2),
                  material->get_as<double>("hardening").value_or(10.0),
                  material->get_as<double>("plast_clamp_lower").value_or(0.975),
                  material->get_as<double>("plast_clamp_higher").value_or(1.0075));
    std::string name = material->get_as<std::string>("name").value_or("");
    material_index[name] = i;
    material_models.push_back(model);
    i++;
  }

  auto objects = config->get_table_array("object");
  i = 0;
  for (const auto &object : *objects) {
    std::cout << "Adding object " << i << "\r" << std::flush;
    u32 material = material_index[object->get_as<std::string>("material").value_or("")];
    auto position = object->get_array_of<double>("position");
    Vec vec_position = toVector(position);
    auto velocity = object->get_array_of<double>("velocity");
    Vec vec_velocity = toVector(velocity);
    simulation.addObject(meshes_dir + object->get_as<std::string>("mesh").value_or("sphere.obj"),
        material,
        object->get_as<double>("size").value_or(1.0),
        vec_position,
        vec_velocity,
        object->get_as<double>("lifetime_begin").value_or(0.0),
        object->get_as<double>("lifetime_end").value_or(std::numeric_limits<double>::max()));
    i++;
  }

  std::cout << "loaded materials" << std::endl;

  simulation.initCuda();

  ParticleWriter writer;

  Renderer renderer(simulation.getFullParticleCount(), flags.save_dir);

  renderer.render(simulation.getActiveParticleList());

  MeshBuilder mesher(simulation.par, flags, flags.mesh_grid);

  bool save = flags.save_dir != "";
  if (save) {
    fs::create_directory(flags.save_dir);
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
  for (u32 i = 0; i < std::numeric_limits<u32>::max(); i++) {
    if (i % 10 == 0) {
      std::cout << "Step " << i << "\r" << std::flush;
    }
    simulation.advance();
    if (i % 20 == 0) {
      simulation.syncDevice();
      renderer.render(simulation.getActiveParticleList());
    }
    if (save && (i % save_every) == 0) {
      std::stringstream ss;
      ss << flags.save_dir << "/meshes/mesh_" << std::setfill('0') << std::setw(5) << frame_id << ".obj";
      mesher.computeMesh(ss.str(), simulation.getActiveParticleList());
      ss.str("");
      ss.clear();
      ss << flags.save_dir << "/particles/particles_" << frame_id << ".bgeo";
      std::string filepath = ss.str();
      writer.writeParticles(filepath, simulation.getActiveParticleList());
      frame_id++;
    }
  }
}
