#ifndef SRC_PLACEMENT_APOLLONIUS_H_

#define SRC_PLACEMENT_APOLLONIUS_H_

#include <thrust/device_vector.h>
#include <memory>
#include "../utils/cuda_array_provider.h"

#define MAX_LABELS 256

/**
 * \brief
 *
 *
 */
class Apollonius
{
 public:
  Apollonius(std::shared_ptr<CudaArrayProvider> inputImage,
             thrust::device_vector<float4> &seedBuffer,
             thrust::device_vector<float> &distances, int labelCount);

  void run();

 private:
  std::shared_ptr<CudaArrayProvider> inputImage;
  thrust::device_vector<float4> &seedBuffer;
  thrust::device_vector<float> &distances;
  thrust::device_vector<int> computeVector;
  thrust::device_vector<int> computeVectorTemp;
  thrust::device_vector<int> seedIds;
  thrust::device_vector<int> seedIndices;
  int labelCount;

  dim3 dimBlock;
  dim3 dimGrid;

  int imageSize;
  int pixelCount;

  cudaSurfaceObject_t outputSurface;

  void resize();
  void runSeedKernel();
  void runStepsKernels();
  void runGatherKernel();
};

#endif  // SRC_PLACEMENT_APOLLONIUS_H_
