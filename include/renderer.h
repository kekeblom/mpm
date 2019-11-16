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
    const u32 frame_width = 800;
    const u32 frame_height = 600;
    const std::string save_dir;
    u8 *frame_buffer;

    void createShaders();
    void createVertices();
	template<class Particle>
    void setParticlePositions(const std::vector<Particle>&);
  public:
  Renderer(const u32, const std::string &);
  ~Renderer();
  template<class Particle>
  void render(const std::vector<Particle>&);
};



template<class Particle>
inline void Renderer::setParticlePositions(const std::vector<Particle> &particles) {
  for (u32 i=0; i < particles.size(); i++) {
    particle_positions[i*3  ] = (particles[i].x[0] - 0.5f) * 1.8f;
    particle_positions[i*3+1] = (particles[i].x[1] - 0.5f) * 1.8f;
    particle_positions[i*3+2] = (particles[i].x[2] - 0.5f) * 1.8f;
  }
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * particles.size(), particle_positions, GL_STREAM_DRAW);
}


template<class Particle>
inline void Renderer::render(const std::vector<Particle> &particles) {
  glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  setParticlePositions(particles);
  glBindVertexArray(vertex_array_object);
  glDrawArrays(GL_POINTS, 0, particles.size());
  if (glGetError()) {
    std::cout << "error";
  }
  glfwSwapBuffers(window);
  glfwPollEvents();
}




