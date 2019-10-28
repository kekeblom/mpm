#include <Eigen/Dense>
#include <cmath>
#include "gtest/gtest.h"
#include "types.h"
#include "linalg.h"

void test_tolerance(const Mat &M) {
  Mat R, S;
  polar_decomposition(M, R, S);
  Mat M2 = R * S;
  float sum = Eigen::abs((M2 - M).array()).sum();
  ASSERT_LT(sum, 1e-5);
}

TEST(TestPolar, Basic) {
  Mat M;
  M << 1, 2, 1, 1, 3, 1, 1, 8, 1;
  test_tolerance(M);
}

TEST(TestPolar, Harder) {
  Mat M;
  M << 0, 1, 0, -1, 2, -1, -1, 0.001, -1;
  test_tolerance(M);
}

TEST(TestPolar, UnitaryHermitian) {
  Mat M, R, S;
  M << 1, 2, 1, 1, 3, 1, 1, 8, 1;
  polar_decomposition(M, R, S);
  // Unitary R.
  float diff = Eigen::abs(((R * R.transpose()) - Mat::Identity()).array()).sum();
  ASSERT_LT(diff, 1e-5);
  // Hermitian S.
  diff = Eigen::abs((S - S.transpose()).array()).sum();
  ASSERT_LT(diff, 1e-5);
}

