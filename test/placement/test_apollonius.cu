#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <vector>
#include "../../src/placement/apollonius.h"
#include "../../src/utils/cuda_helper.h"
#include "../cuda_array_mapper.h"

std::vector<int> callApollonoius(std::vector<Eigen::Vector4f> &image,
                                 std::vector<float> distances, int imageSize,
                                 std::vector<Eigen::Vector4f> labelsSeed)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  int labelCount = labelsSeed.size();
  auto imageMapper = std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(
      imageSize, imageSize, image, channelDesc);

  cudaChannelFormatDesc channelDescDistances =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  auto distancesMapper = std::make_shared<CudaArrayMapper<float>>(
      imageSize, imageSize, distances, channelDescDistances);


  Apollonius apollonius(distancesMapper, imageMapper, labelsSeed, labelCount);
  apollonius.run();

  image = imageMapper->copyDataFromGpu();

  imageMapper->unmap();

  std::vector<int> result;
  thrust::host_vector<int> labelIndices = apollonius.getIds();
  for (auto index : labelIndices)
    result.push_back(index);

  return result;
}

