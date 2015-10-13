#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <vector>
#include "../../src/placement/apollonius.h"
#include "../../src/utils/cuda_helper.h"
#include "../cuda_array_mapper.h"

void callApollonoius(std::vector<Eigen::Vector4f> &image,
                     std::vector<float> distances)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  int labelCount = 1;
  int imageSize = 4;
  int pixelCount = imageSize * imageSize;
  auto imageMapper = std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(
      imageSize, imageSize, image, channelDesc);
  imageMapper->map();

  thrust::device_vector<float4> seedBuffer(pixelCount, make_float4(0, 0, 0, 0));
  seedBuffer[0] = make_float4(1, 1, 1, 1);
  thrust::device_vector<float> distanceVector(distances);
  thrust::device_vector<int> computeVector;
  thrust::device_vector<int> computeVectorTemp;
  thrust::device_vector<int> computeSeedIds;
  thrust::device_vector<int> computeSeedIndices;

  cudaJFAApolloniusThrust(imageMapper->getArray(), imageSize, labelCount,
                          seedBuffer, distanceVector, computeVector,
                          computeVectorTemp, computeSeedIds,
                          computeSeedIndices);

  image = imageMapper->copyDataFromGpu();

  imageMapper->unmap();
}

