#ifndef TEST_CUDA_ARRAY_3D_MAPPER_H_

#define TEST_CUDA_ARRAY_3D_MAPPER_H_

#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include "../src/utils/cuda_helper.h"
#include "../src/utils/cuda_array_provider.h"

/**
 * \brief CudaArrayProvider implementation for tests, taking data from a given
 * vector
 *
 * The data can be updated using #updateData. It sets the data internally, but
 * does not copy it to the GPU. This can be done by calling #map (which might
 * be called be the tested code anyway).
 *
 * Processed data can be retrieved from the GPU by calling #copyDataFromGpu.
 */
template <class ElementType> class CudaArray3DMapper : public CudaArrayProvider
{
 public:
  CudaArray3DMapper(int width, int height, int depth,
                    std::vector<ElementType> data,
                    cudaChannelFormatDesc channelFormat,
                    unsigned int flags = cudaArraySurfaceLoadStore)
    : CudaArrayProvider(width, height), data(data),
      channelFormat(channelFormat), flags(flags)
  {
    this->depth = depth;
  }

  virtual void map()
  {
    cudaExtent extent = make_cudaExtent(width, height, depth);
    if (!array)
      HANDLE_ERROR(cudaMalloc3DArray(&array, &channelFormat, extent, flags));

    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr(
        data.data(), width * sizeof(ElementType), width, height);
    copyParams.dstArray = array;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;

    HANDLE_ERROR(cudaMemcpy3D(&copyParams));
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

#endif  // TEST_CUDA_ARRAY_3D_MAPPER_H_
