#ifndef TEST_CUDA_ARRAY_MAPPER_H_

#define TEST_CUDA_ARRAY_MAPPER_H_

#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include "../src/utils/cuda_helper.h"
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
                  cudaChannelFormatDesc channelFormat,
                  unsigned int flags = cudaArraySurfaceLoadStore)
    : CudaArrayProvider(width, height), data(data),
      channelFormat(channelFormat), flags(flags)
  {
  }

  virtual void map()
  {
    HANDLE_ERROR(cudaMallocArray(&array, &channelFormat, width, height, flags));

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
    assert(width * height == result.size());
    HANDLE_ERROR(cudaMemcpyFromArray(result.data(), array, 0, 0,
                                     sizeof(ElementType) * width * height,
                                     cudaMemcpyDeviceToHost));

    return result;
  }

 private:
  std::vector<ElementType> data;
  cudaArray_t array;
  cudaChannelFormatDesc channelFormat;
  unsigned int flags;
};

#endif  // TEST_CUDA_ARRAY_MAPPER_H_
