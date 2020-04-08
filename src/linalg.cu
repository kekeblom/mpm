#include <Eigen/Dense>
#include "linalg.h"
#include "svd3_cuda.h"

namespace linalg {

void polar_decomposition(const Mat &A, Mat &R, Mat &S) {
  //NOTE: a faster algorithm exists. See N. Higham 2015.
  auto svd = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto singularValues = svd.singularValues();
  auto U = svd.matrixU();
  auto V = svd.matrixV();
  auto V_T = V.transpose();
  R = U * V_T;
  S = V * singularValues.asDiagonal() * V_T;
}

__device__ void polar_decomposition_device(const Mat& A, Mat& R, Mat& S) {
  Mat U, Z(Mat::Zero()), V;
  svd(A(0, 0), A(0, 1), A(0, 2),
      A(1, 0), A(1, 1), A(1, 2),
      A(2, 0), A(2, 1), A(2, 2),
      U(0, 0), U(0, 1), U(0, 2),
      U(1, 0), U(1, 1), U(1, 2),
      U(2, 0), U(2, 1), U(2, 2),
      Z(0, 0), Z(1, 1), Z(2, 2),
      V(0, 0), V(0, 1), V(0, 2),
      V(1, 0), V(1, 1), V(1, 2),
      V(2, 0), V(2, 1), V(2, 2));
  Mat V_T = V.transpose();
  R = U * V_T;
  S = V * Z * V_T;
}

__device__ void svd_decomposition(const Mat& A, Mat& U, Mat& S, Mat& V) {
  svd(A(0, 0), A(0, 1), A(0, 2),
      A(1, 0), A(1, 1), A(1, 2),
      A(2, 0), A(2, 1), A(2, 2),
      U(0, 0), U(0, 1), U(0, 2),
      U(1, 0), U(1, 1), U(1, 2),
      U(2, 0), U(2, 1), U(2, 2),
      S(0, 0), S(1, 1), S(2, 2),
      V(0, 0), V(0, 1), V(0, 2),
      V(1, 0), V(1, 1), V(1, 2),
      V(2, 0), V(2, 1), V(2, 2));
}

__device__ real determinant(const Mat& M) {
  real sub1 = M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0);
  real sub2 = M(1, 0) * M(2, 2) - M(1, 2) * M(2, 0);
  real sub3 = M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1);
  return M(0, 0) * sub3 - M(0, 1) * sub2 + M(0, 2) * sub1;
}
}
