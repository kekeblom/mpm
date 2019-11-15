#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <boost/filesystem.hpp>
#include "renderer.h"

namespace fs = boost::filesystem;

const float TriangleBase = 0.02;
const float TriangleAngle = M_PI / 2.0 / 3.0 / 2.0; // 30 degrees.
const float TriangleHeight = 0.5 * TriangleBase / std::tan(TriangleAngle);

const char *VertexShaderSource =
#include "cube_vertex.glsl"
;
const char *FragmentShaderSource =
#include "constant_fragment.glsl"
;

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
}

void Renderer::createShaders() {
  unsigned int vertex_shader, fragment_shader;
  vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &VertexShaderSource, NULL);
  glCompileShader(vertex_shader);
  int success;
  char log[512];
  glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertex_shader, 512, NULL, log);
    std::cout << "Shader compilation failed: " << log << "\n";
  }

  fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &FragmentShaderSource, NULL);
  glCompileShader(fragment_shader);

  glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragment_shader, 512, NULL, log);
    std::cout << "Shader compilation failed " << log << "\n";
  }

  shader_program = glCreateProgram();
  glAttachShader(shader_program, vertex_shader);
  glAttachShader(shader_program, fragment_shader);
  glLinkProgram(shader_program);

  glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shader_program, 512, NULL, log);
    std::cout << "Shader linking failed " << log << "\n";
  }

  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);
}

void Renderer::createVertices() {
  glGenVertexArrays(1, &vertex_array_object);
  glBindVertexArray(vertex_array_object);
  glGenBuffers(1, &vertex_buffer_object);

  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * particle_count, particle_positions, GL_STREAM_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::setParticlePositions(const std::vector<Particle> &particles) {
  for (u32 i=0; i < particle_count; i++) {
    particle_positions[i*3  ] = (particles[i].x[0] - 0.5f) * 1.8f;
    particle_positions[i*3+1] = (particles[i].x[1] - 0.5f) * 1.8f;
    particle_positions[i*3+2] = (particles[i].x[2] - 0.5f) * 1.8f;
  }
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * particle_count, particle_positions, GL_STREAM_DRAW);
}

Renderer::Renderer(const u32 num_particles, const std::string &save_dir) : save_dir(save_dir) {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  window = glfwCreateWindow(800, 600, "Simulation", NULL, NULL);
  if (window == NULL) {
      std::cout << "Failed to create glfw window\n";
      glfwTerminate();
      return;
  }
  glfwMakeContextCurrent(window);
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return;
  glViewport(0, 0, frame_width, frame_height);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

  particle_positions = new float[num_particles * 3];
  particle_count = num_particles;

  createShaders();
  glUseProgram(shader_program);
  createVertices();

  frame_buffer = new u8[frame_width * frame_height * 3];
}

Renderer::~Renderer() {
  delete[] particle_positions;
  glfwTerminate();
  glDeleteBuffers(1, &vertex_buffer_object);
  glDeleteVertexArrays(1, &vertex_array_object);
  delete[] frame_buffer;
}

void Renderer::render(const std::vector<Particle> &particles) {
  glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  setParticlePositions(particles);
  glBindVertexArray(vertex_array_object);
  glDrawArrays(GL_POINTS, 0, particle_count);
  if (glGetError()) {
    std::cout << "error";
  }
  glfwSwapBuffers(window);
  glfwPollEvents();
}

