#ifndef H_LINALG
#define H_LINALG
#include <Eigen/Dense>
#include "types.h"

void polar_decomposition(const Mat &, Mat &, Mat &);
#ifdef __CUDACC__
__device__ void polar_decomposition_device(const Mat &, Mat &, Mat &);
#endif

#endif H_LINALG
