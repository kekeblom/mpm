#include <eigen3/Eigen/Dense>
#ifndef COMMON_TYPES
#define COMMON_TYPES

using real = float;
using u8 = uint8_t;
using u32 = unsigned int;
using u64 = unsigned long;
using i32 = int;
using i64 = long;
using f32 = float;
using f64 = double;

using Vec = Eigen::Matrix<real, 3, 1>;
using Mat = Eigen::Matrix<real, 3, 3>;

using Veci = Eigen::Matrix<int, 3, 1>;
using VecU32 = Eigen::Matrix<u32, 3, 1>;


struct Particle {
  Vec x; // Position.
  Vec v; // Velocity.
  Mat F; // Deformation gradient.
  Mat C; // Affine momentum.
  real Jp; // Determinant of the deformation gradient.
  Particle(Vec &x, Vec v = Vec::Zero()) :
    x(x[0], x[1], x[2]),
    v(v[0], v[1], v[2]),
    F(Mat::Identity()),
    C(Mat::Zero()),
    Jp(1.0)
    {}
};

#endif
