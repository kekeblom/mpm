#include <Eigen/Dense>
#ifndef COMMON_TYPES
#define COMMON_TYPES

using real = float;
using u8 = uint8_t;
using u32 = uint32_t;//unsigned int;
using u64 = uint64_t;//unsigned long;
using i32 = int32_t;//int;
using i64 = int64_t;//long;
using f32 = float;
using f64 = double;

using Vec = Eigen::Matrix<real, 3, 1>;
using Mat = Eigen::Matrix<real, 3, 3>;

using Vec4 = Eigen::Matrix<real, 4, 1>;

using Veci = Eigen::Matrix<int, 3, 1>;
using Vecu32 = Eigen::Matrix<u32, 3, 1>;


struct ParticleBase {
    // minimal form of particle.
    // Extended particle types needed for more elaborate transfer schemes would derive from here.

    u8 material_type;
    Vec x; // Position.
    Vec v; // Velocity.
    Mat F; // Deformation gradient.

    ParticleBase(int type = 0, Vec x = Vec::Zero(), Vec v = Vec::Zero())
	: material_type(type), x(x), v(v), F(Mat::Identity()) {}
};


#endif
