#ifndef SRC_UTILS_CUDA_HELPER_H_

#define SRC_UTILS_CUDA_HELPER_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#define HANDLE_ERROR(error)                                                    \
  {                                                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
      printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__,     \
             __LINE__);                                                        \
      throw std::runtime_error(cudaGetErrorString(error));                     \
    }                                                                          \
  }

#define HANDLE_NULL(a)                                                         \
  {                                                                            \
    if (a == NULL)                                                             \
    {                                                                          \
      printf("Host memory failed in %s at line %d\n", __FILE__, __LINE__);     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#endif  // SRC_UTILS_CUDA_HELPER_H_
