#include "types.h"
#include "mpm.h"
#include "mesh_builder.h"
#include "particle_writer.h"

const unsigned int FrameRate = 240; // for export of data
std::vector<MaterialModel> material_models;

int main(int argc, char *argv[]) {
  CLIOptions flags(argc, argv);

  Simulation simulation(flags, InterpolationKernel(), material_models);

  // "Scene selector"
  // usage: uncomment the desired scene below ;)
  // (also, set the correct path to the mesh directory)
  std::string meshes_dir = "../meshes/";

  ////////////////////////////
  // Scene: Snowman gets hit (by snowball)
  ////////////////////////////
  material_models.push_back(MaterialModel(1.0/flags.particle_count, 700, 1.4e5, 0.2, 10, 1.0-2.5e-2, 1.0+0.75e-2)); // soft, light snow
  simulation.addObject(meshes_dir + "sphere.obj",
                       0,
                       0.4,
                       Vec(0.3, 0.03, 0.3),
                       Vec(0.0,0.0,0.0));
  simulation.addObject(meshes_dir + "sphere.obj",
                       0,
                       0.3,
                       Vec(0.35, 0.43, 0.35),
                       Vec(0.0,0.0,0.0));
  simulation.addObject(meshes_dir + "sphere.obj",
                       0,
                       0.2,
                       Vec(0.4, 0.75, 0.4),
                       Vec(0.0,0.0,0.0));
  simulation.addObject(meshes_dir + "sphere.obj",
                       0,
                       0.1,
                       Vec(0.05, 0.5, 0.46),
                       Vec(7.0,2.5,0.0),
                       0.5);

  simulation.initCuda();
  ////////////////////////////
  // Scene: Rubber duck gets hit (by metal cube)
  ////////////////////////////
//  simulation.addObject(meshes_dir + "rubber_duck.obj",
//                       MaterialModel(1.0/flags.particle_count, 200, 1.4e5, 0.2, 0, 0.0, 1.0e30),
//                       0.4,
//                       Vec(0.3, 0.4, 0.3),
//                       Vec(0.0,0.0,0.0));
//  simulation.addObject(meshes_dir + "cube.obj",
//                       MaterialModel(1.0/flags.particle_count, 8000, 5e7, 0.45, 0, 0.0, 1.0e30), // "Fluid"????
//                       0.3,
//                       Vec(0.35, 0.5, 0.35),
//                       Vec(0.0,0.0,0.0),
//                       2.0);

  ////////////////////////////
  // Scene: Liquid bunny
  ////////////////////////////
//  simulation.addObject(meshes_dir + "stanford_bunny.obj",
//                       MaterialModel(1.0/flags.particle_count, 1000, 1.4e5, 0.45, 0, 0.975, 0.975),
//                       0.5,
//                       Vec(0.25, 0.4, 0.25),
//                       Vec(0.0,0.0,0.0),
//                       0.0);

  ////////////////////////////
  // Scene: Marshmallow duck and water
  ////////////////////////////
//  simulation.addObject(meshes_dir + "rubber_duck.obj",
//                       MaterialModel(1.0/flags.particle_count, 200, 1.4e5, 0.2, 0, 0.0, 1.0e30),
//                       0.4,
//                       Vec(0.3, 0.4, 0.3),
//                       Vec(0.0,0.0,0.0));
//  simulation.addObject(meshes_dir + "cube.obj",
//                       MaterialModel(1.0/flags.particle_count, 1000, 1.4e5, 0.45, 0, 0.975, 0.975), // "Fluid"????
//                       0.35,
//                       Vec(0.325, 0.5, 0.325),
//                       Vec(0.0,0.0,0.0),
//                       1.8);


  ////////////////////////////
  // General Examples with mesh from command line arguments
  ////////////////////////////
//  simulation.addObject(flags.load_mesh,
//                       MaterialModel(1.0/flags.particle_count, 400, 1.4e5, 0.2, 10, 1.0-1e-2, 1.0+3e-3),  // soft, light snow
//                       0.3,
//                       Vec(0.3, 0.6, 0.3),
//                       Vec(0.0,0.0,0.0));

//  simulation.addObject(flags.load_mesh,
//                       MaterialModel(1.0/flags.particle_count, 700, 1.4e5, 0.2, 10, 1.0-5e-2, 1.0+1.5e-2),  // harder, heavier snow
//                       0.2,
//                       Vec(0.35, 0.1, 0.3),
//                       Vec(0.0,5.0,0.0));

//  simulation.addObject(flags.load_mesh,
//                       MaterialModel(1.0/flags.particle_count, 400, 1.4e5, 0.2, 0, 0.0, 1.0e30),  // "Jelly", by abuse of the snow material model
//                       0.3,
//                       Vec(0.3, 0.6, 0.3),
//                       Vec(0.0,0.0,0.0));

//  simulation.addObject(flags.load_mesh,
//                       MaterialModel(1.0/flags.particle_count, 1000, 1.4e5, 0.45, 0, 0.97, 0.97), // "Fluid"????
//                       0.5,
//                       Vec(0.25, 0.4, 0.25),
//                       Vec(0.0,0.0,0.0));


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
  for (unsigned int i = 0; i < std::numeric_limits<unsigned int>::max(); i++) {
    std::cout << "Step " << i << "\r" << std::flush;
    simulation.advance();
    if(i%20 == 0) {
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
