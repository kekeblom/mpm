#include <Eigen/Core>
#include <cmath>
#include "gtest/gtest.h"
#include "types.h"
#include "linalg.h"
#include "gpu.h"

using namespace linalg;

void test_tolerance(const Mat &M, const Mat &R, const Mat &S) {
  Mat M2 = R * S;
  float sum = Eigen::abs((M2 - M).array()).sum();
  ASSERT_LT(sum, 1e-5);
}

void test_tolerance(const Mat &M) {
  Mat R, S;
  polar_decomposition(M, R, S);
  test_tolerance(M, R, S);
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

__global__ void polarDecompose(const Mat* A, Mat* R, Mat* S) {
  linalg::polar_decomposition_device(*A, *R, *S);
}

__global__ void computeDeterminant(const Mat& A, real *out) {
  *out = determinant(A);
}

TEST(TestDevicePolar, Basic) {
  Mat M, R, S;
  M << 1, 2, 1, 1, 3, 1, 1, 8, 1;
  Mat * device_M, *device_R, *device_S;
  cudaMalloc((void **)&device_M, sizeof(Mat));
  cudaMalloc((void **)&device_R, sizeof(Mat));
  cudaMalloc((void **)&device_S, sizeof(Mat));
  cudaMemcpy(device_M, &M, sizeof(Mat), cudaMemcpyHostToDevice);
  polarDecompose<<<1, 1>>>(device_M, device_R, device_S);
  checkGpuError(cudaGetLastError());
  checkGpuError(cudaDeviceSynchronize());
  cudaMemcpy(&R, device_R, sizeof(Mat), cudaMemcpyDeviceToHost);
  cudaMemcpy(&S, device_S, sizeof(Mat), cudaMemcpyDeviceToHost);
  test_tolerance(M, R, S);

  cudaFree(device_M);
  cudaFree(device_R);
  cudaFree(device_S);
}

TEST(TestDeterminant, Identity) {
  Mat M = Mat::Identity();
  Mat * device_M;
  real det;
  real* device_out;
  cudaMalloc((void **)&device_M, sizeof(Mat));
  cudaMalloc((void **)&device_out, sizeof(real));
  cudaMemcpy(device_M, &M, sizeof(Mat), cudaMemcpyHostToDevice);
  computeDeterminant<<<1, 1>>>(*device_M, device_out);
  cudaMemcpy(&det, device_out, sizeof(real), cudaMemcpyDeviceToHost);
  checkGpuError(cudaGetLastError());
  checkGpuError(cudaDeviceSynchronize());
  ASSERT_EQ(det, 1.0);
  cudaFree(device_M);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

