#include <Eigen/Dense>
#include "linalg.h"

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

