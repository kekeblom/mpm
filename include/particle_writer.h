#ifdef PARTIO_AVAILABLE
#include <Partio.h>
#endif

#include "types.h"

class ParticleWriter {

  public:
  void writeParticles(const std::string&, const std::vector<Particle>&);
};

