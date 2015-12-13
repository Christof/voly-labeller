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
    assert(width * height == static_cast<int>(result.size()));
    HANDLE_ERROR(cudaMemcpyFromArray(result.data(), array, 0, 0,
                                     sizeof(ElementType) * width * height,
                                     cudaMemcpyDeviceToHost));

    return result;
  }

  void updateData(std::vector<ElementType> newData)
  {
    assert(width * height == static_cast<int>(newData.size()));
    data = newData;
  }

  ElementType getDataAt(int index)
  {
    return data[index];
  }

 private:
  std::vector<ElementType> data;
  cudaArray_t array = nullptr;
  cudaChannelFormatDesc channelFormat;
  unsigned int flags;
};

#endif  // TEST_CUDA_ARRAY_MAPPER_H_
