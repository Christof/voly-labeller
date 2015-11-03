#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <vector>
#include "../../src/placement/apollonius.h"
#include "../../src/utils/cuda_helper.h"
#include "../cuda_array_mapper.h"

std::vector<int> callApollonoius(std::vector<Eigen::Vector4f> &image,
                     std::vector<float> distances)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  int labelCount = 1;
  int imageSize = 4;
  auto imageMapper = std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(
      imageSize, imageSize, image, channelDesc);

  cudaChannelFormatDesc channelDescDistances =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  auto distancesMapper = std::make_shared<CudaArrayMapper<float>>(
      imageSize, imageSize, distances, channelDescDistances);
  thrust::device_vector<float4> seedBuffer(labelCount, make_float4(0, 0, 0, 0));
  seedBuffer[0] = make_float4(0, 2, 1, 1);

  Apollonius apollonius(distancesMapper, imageMapper, seedBuffer, labelCount);
  apollonius.run();

  image = imageMapper->copyDataFromGpu();

  imageMapper->unmap();

  std::vector<int> result;
  thrust::host_vector<int> labelIndices = apollonius.getIds();
  for (auto index : labelIndices)
    result.push_back(index);

  return result;
}

