#ifndef SRC_UTILS_CUDA_HELPER_H_

#define SRC_UTILS_CUDA_HELPER_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

static inline void HandleError(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess)
  {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

#define HANDLE_ERROR(error) HandleError(error, __FILE__, __LINE__)

inline unsigned int divUp(unsigned int a, unsigned int b)
{
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

#endif  // SRC_UTILS_CUDA_HELPER_H_
