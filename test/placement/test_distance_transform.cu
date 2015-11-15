#include <thrust/device_vector.h>
#include <vector>
#include "../../src/placement/distance_transform.h"
#include "../../src/utils/cuda_helper.h"
#include "../cuda_array_mapper.h"

std::vector<float> callDistanceTransform(
    std::shared_ptr<CudaArrayMapper<float>> occupancyImageProvider,
    std::vector<float> &result)
{
  cudaChannelFormatDesc outputChannelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  int pixelCount =
      occupancyImageProvider->getWidth() * occupancyImageProvider->getHeight();
  std::vector<float> resultImage(pixelCount);
  auto output = std::make_shared<CudaArrayMapper<float>>(
      occupancyImageProvider->getWidth(), occupancyImageProvider->getHeight(),
      resultImage, outputChannelDesc);

  DistanceTransform distanceTransform(occupancyImageProvider, output);
  distanceTransform.run();

  thrust::host_vector<float> resultHost = distanceTransform.getResults();
  for (auto element : resultHost)
    result.push_back(element);

  resultImage = output->copyDataFromGpu();

  output->unmap();

  return resultImage;
}

