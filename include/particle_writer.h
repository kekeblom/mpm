#ifdef PARTIO_AVAILABLE
#include <Partio.h>
#endif

#include "types.h"



class ParticleWriter {

  public:
  template<class Particle>
  void writeParticles(const std::string&, const std::vector<Particle>&);
};




template<class Particle>
inline void ParticleWriter::writeParticles(const std::string& filepath, const std::vector<Particle>&particles) {
#ifdef PARTIO_AVAILABLE
  Partio::ParticlesDataMutable &particle_data = *Partio::create();
  Partio::ParticleAttribute id_attr = particle_data.addAttribute("id", Partio::INT, 1);
  Partio::ParticleAttribute position_attr = particle_data.addAttribute("position", Partio::VECTOR, 3);
  Partio::ParticleAttribute velocity_attr = particle_data.addAttribute("velocity", Partio::VECTOR, 3);
  Partio::ParticleAttribute radius_attr = particle_data.addAttribute("radius", Partio::FLOAT, 1);
  int i = 0;
  for (auto particle : particles) {
    Partio::ParticleIndex index = particle_data.addParticle();
    int *id = particle_data.dataWrite<int>(id_attr, index);
    float *position = particle_data.dataWrite<float>(position_attr, index);
    float *velocity = particle_data.dataWrite<float>(velocity_attr, index);
    float *radius = particle_data.dataWrite<float>(radius_attr, index);
    radius[0] = 0.1;

    id[0] = i;
    i++;
    position[0] = particle.x(0);
    position[1] = particle.x(1);
    position[2] = particle.x(2);
    velocity[0] = particle.v(0);
    velocity[1] = particle.v(1);
    velocity[2] = particle.v(2);
  }
  Partio::write(filepath.c_str(), particle_data, true);
  particle_data.release();
#else
	std::cout << "Warning: Particles could not be written because Partio library is missing." << std::endl;
#endif
}
