#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <vector>
#include "../../src/placement/distance_transform.h"
#include "../../src/utils/cuda_helper.h"

void callApollonoius(std::vector<Eigen::Vector4f> &image)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaArray_t array;
  int labelCount = 1;
  int imageSize = 4;
  int pixelCount = imageSize * imageSize;
  HANDLE_ERROR(cudaMallocArray(&array, &channelDesc, imageSize, imageSize,
                               cudaArraySurfaceLoadStore));
  HANDLE_ERROR(cudaMemcpyToArray(array, 0, 0, image.data(),
                                 pixelCount * sizeof(Eigen::Vector4f),
                                 cudaMemcpyHostToDevice));

  thrust::device_vector<float4> seedBuffer(pixelCount, make_float4(0, 0, 0, 0));
  seedBuffer[0] = make_float4(1, 1, 1, 1);
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

std::vector<Eigen::Vector4f> callDistanceTransform(std::vector<float> depth,
    std::vector<float> &result)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray_t array;
  int imageSize = 4;
  int pixelCount = imageSize * imageSize;
  HANDLE_ERROR(cudaMallocArray(&array, &channelDesc, imageSize, imageSize));
  HANDLE_ERROR(cudaMemcpyToArray(array, 0, 0, depth.data(),
                                 pixelCount * sizeof(float),
                                 cudaMemcpyHostToDevice));

  cudaArray_t outputArray;
  cudaChannelFormatDesc outputChannelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  HANDLE_ERROR(cudaMallocArray(&outputArray, &outputChannelDesc, imageSize,
                               imageSize, cudaArraySurfaceLoadStore));

  thrust::device_vector<int> computeVector;
  thrust::device_vector<float> resultVector;

  cudaJFADistanceTransformThrust(array, channelDesc, outputArray, imageSize,
                                 imageSize, imageSize, computeVector,
                                 resultVector);

  std::vector<Eigen::Vector4f> resultImage(pixelCount);
  HANDLE_ERROR(cudaMemcpyFromArray(resultImage.data(), outputArray, 0, 0,
                                   pixelCount * sizeof(Eigen::Vector4f),
                                   cudaMemcpyDeviceToHost));

  cudaFree(array);
  cudaFree(outputArray);

  thrust::host_vector<float> resultHost = resultVector;
  for (auto element : resultHost)
    result.push_back(element);

  return resultImage;
}

