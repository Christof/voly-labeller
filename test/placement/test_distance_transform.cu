#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <vector>
#include "../../src/placement/distance_transform.h"
#include "../../src/utils/cuda_helper.h"

void callDistanceTransform(std::vector<Eigen::Vector4f> &image)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaArray_t array;
  int labelCount = 1;
  int imageSize = 4;
  int pixelCount = imageSize * imageSize;
  HANDLE_ERROR(cudaMallocArray(&array, &channelDesc, imageSize, imageSize));
  HANDLE_ERROR(cudaMemcpyToArray(array, 0, 0, image.data(),
                                 pixelCount * sizeof(Eigen::Vector4f),
                                 cudaMemcpyHostToDevice));

  thrust::device_vector<float4> seedBuffer(pixelCount * 4,
                                           make_float4(0, 0, 0, 0));
  thrust::device_vector<float> distanceVector;
  thrust::device_vector<int> computeVector;
  thrust::device_vector<int> computeVectorTemp;
  thrust::device_vector<int> computeSeedIds;
  thrust::device_vector<int> computeSeedIndices;

  cudaJFAApolloniusThrust(array, imageSize, labelCount, seedBuffer,
                          distanceVector, computeVector, computeVectorTemp,
                          computeSeedIds, computeSeedIndices);

  HANDLE_ERROR(cudaMemcpyFromArray(image.data(), array, 0, 0,
                                   pixelCount * sizeof(Eigen::Vector4f),
                                   cudaMemcpyDeviceToHost));

  cudaFree(array);
}
