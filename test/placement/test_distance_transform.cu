#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <vector>
#include "../../src/placement/distance_transform.h"
#include "../../src/utils/cuda_helper.h"
#include "../cuda_array_mapper.h"

std::vector<Eigen::Vector4f> callDistanceTransform(
    std::shared_ptr<CudaArrayMapper<float>> depthImageProvider,
    std::vector<float> &result)
{
  cudaChannelFormatDesc outputChannelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  int pixelCount =
      depthImageProvider->getWidth() * depthImageProvider->getHeight();
  std::vector<Eigen::Vector4f> resultImage(pixelCount);
  auto output = std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(
      depthImageProvider->getWidth(), depthImageProvider->getHeight(),
      resultImage, outputChannelDesc);

  DistanceTransform distanceTransform(depthImageProvider, output);
  distanceTransform.run();

  thrust::host_vector<float> resultHost = distanceTransform.getResults();
  for (auto element : resultHost)
    result.push_back(element);

  resultImage = output->copyDataFromGpu();

  output->unmap();

  return resultImage;
}

