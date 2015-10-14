#ifndef SRC_PLACEMENT_APOLLONIUS_H_

#define SRC_PLACEMENT_APOLLONIUS_H_

#include <memory>
#include <thrust/device_vector.h>
#include "../utils/cuda_array_provider.h"

#define MAX_LABELS 256

void cudaJFAApolloniusThrust(cudaArray_t imageArray, int imageSize,
                             int numLabels,
                             thrust::device_vector<float4> &seedbuffer,
                             thrust::device_vector<float> &distance_vector,
                             thrust::device_vector<int> &compute_vector,
                             thrust::device_vector<int> &compute_temp_vector,
                             thrust::device_vector<int> &compute_seed_ids,
                             thrust::device_vector<int> &compute_seed_indices);

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
             thrust::device_vector<float> &distances,
             int numLabels);

  void run();

 private:
  std::shared_ptr<CudaArrayProvider> inputImage;
  thrust::device_vector<float4> &seedBuffer;
  thrust::device_vector<float> &distances;
  thrust::device_vector<int> computeVector;
  thrust::device_vector<int> computeVectorTemp;
  thrust::device_vector<int> seedIds;
  thrust::device_vector<int> seedIndices;
  int numLabels;

  dim3 dimBlock;
  dim3 dimGrid;

  int imageSize;
  int pixelCount;

  void resize();
  void runSeedKernel();
  void runStepsKernels();
  void runGatherKernel();
};

#endif  // SRC_PLACEMENT_APOLLONIUS_H_
