#ifndef SRC_UTILS_CUDA_ARRAY_PROVIDER_H_

#define SRC_UTILS_CUDA_ARRAY_PROVIDER_H_

#include <cuda_runtime.h>

/**
 * \brief Interface for classes which provide access to a cudaArray
 *
 */
class CudaArrayProvider
{
 public:
  virtual void map() = 0;
  virtual void unmap() = 0;

  virtual cudaChannelFormatDesc getChannelDesc() = 0;
  virtual cudaArray_t getArray() = 0;
};

#endif  // SRC_UTILS_CUDA_ARRAY_PROVIDER_H_
