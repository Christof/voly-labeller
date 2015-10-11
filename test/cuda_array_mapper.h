#ifndef TEST_CUDA_ARRAY_MAPPER_H_

#define TEST_CUDA_ARRAY_MAPPER_H_

#include <vector>
#include <cuda_runtime.h>
#include "../src/utils/cuda_array_provider.h"

/**
 * \brief
 *
 *
 */

template <class ElementType> class CudaArrayMapper : public CudaArrayProvider
{
 public:
  CudaArrayMapper(int width, int height, std::vector<ElementType> data,
                  cudaChannelFormatDesc channelFormat)
    : width(width), height(height), data(data), channelFormat(channelFormat)
  {
  }

  virtual void map()
  {
    HANDLE_ERROR(cudaMallocArray(&array, &channelFormat, width, height,
                                 cudaArraySurfaceLoadStore));
    HANDLE_ERROR(cudaMemcpyToArray(array, 0, 0, data.data(),
                                   sizeof(ElementType) * data.size(),
                                   cudaMemcpyHostToDevice));
  }
  virtual void unmap()
  {
    cudaFree(array);
  }

  virtual cudaChannelFormatDesc getChannelDesc()
  {
    return channelFormat;
  }

  virtual cudaArray_t getArray()
  {
    return array;
  }

  std::vector<ElementType> copyDataFromGpu()
  {
    std::vector<ElementType> result(width * height);
    HANDLE_ERROR(cudaMemcpyFromArray(result.data(), array, 0, 0,
                                     sizeof(ElementType) * width * height,
                                     cudaMemcpyDeviceToHost));

    return result;
  }

 private:
  int width;
  int height;
  std::vector<ElementType> data;
  cudaArray_t array;
  cudaChannelFormatDesc channelFormat;
};

#endif  // TEST_CUDA_ARRAY_MAPPER_H_