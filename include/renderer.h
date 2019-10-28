#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "types.h"

class Renderer {
  private:
    GLFWwindow *window;
    float *particle_positions;
    u32 shader_program;

    u32 vertex_array_object;
    u32 vertex_buffer_object;

    u32 particle_count;
    u32 frame_index = 0;
    const u32 frame_width = 800;
    const u32 frame_height = 600;
    const std::string save_dir;
    u8 *frame_buffer;

    void createShaders();
    void createVertices();
    void setParticlePositions(const std::vector<Particle>&);
  public:
  Renderer(const u32, const std::string &);
  ~Renderer();
  void writeFrame();
  void render(const std::vector<Particle>&);
};

