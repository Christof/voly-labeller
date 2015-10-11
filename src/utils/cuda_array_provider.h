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
  CudaArrayProvider(int width, int height) : width(width), height(height)
  {
  }

  virtual void map() = 0;
  virtual void unmap() = 0;

  virtual cudaChannelFormatDesc getChannelDesc() = 0;
  virtual cudaArray_t getArray() = 0;

  int getWidth()
  {
    return width;
  }

  int getHeight()
  {
    return height;
  }

 protected:
  int width;
  int height;
};

#endif  // SRC_UTILS_CUDA_ARRAY_PROVIDER_H_
