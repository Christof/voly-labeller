#include "../test.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include "../../src/placement/distance_transform.h"
#include "../../src/utils/cuda_helper.h"
#include "../cuda_array_mapper.h"

std::vector<float> callDistanceTransform(
    std::shared_ptr<CudaArrayMapper<float>> occupancyImageProvider)
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

  resultImage = output->copyDataFromGpu();

  output->unmap();

  return resultImage;
}

TEST(Test_DistanceTransform, DistanceTransform)
{
  std::vector<float> occupancyImage;

  occupancyImage.push_back(0.0f);
  occupancyImage.push_back(0.0f);
  occupancyImage.push_back(0.0f);
  occupancyImage.push_back(0.0f);

  occupancyImage.push_back(0.0f);
  occupancyImage.push_back(1.0f);
  occupancyImage.push_back(1.0f);
  occupancyImage.push_back(0.0f);

  occupancyImage.push_back(0.0f);
  occupancyImage.push_back(1.0f);
  occupancyImage.push_back(1.0f);
  occupancyImage.push_back(0.0f);

  occupancyImage.push_back(0.0f);
  occupancyImage.push_back(0.0f);
  occupancyImage.push_back(0.0f);
  occupancyImage.push_back(0.0f);

  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  auto occupancyImageProvider =
      std::make_shared<CudaArrayMapper<float>>(4, 4, occupancyImage, channelDesc);
  auto image = callDistanceTransform(occupancyImageProvider);

  EXPECT_NEAR(0.0f, image[0], 1e-4f);
  EXPECT_NEAR(0.0f, image[1], 1e-4f);
  EXPECT_NEAR(0.0f, image[2], 1e-4f);
  EXPECT_NEAR(0.0f, image[3], 1e-4f);

  EXPECT_NEAR(0.0f, image[4], 1e-4f);
  EXPECT_NEAR(1.0f / 32, image[5], 1e-4f);
  EXPECT_NEAR(1.0f / 32, image[6], 1e-4f);
  EXPECT_NEAR(0.0f, image[7], 1e-4f);

  EXPECT_NEAR(0.0f, image[8], 1e-4f);
  EXPECT_NEAR(1.0f / 32, image[9], 1e-4f);
  EXPECT_NEAR(1.0f / 32, image[10], 1e-4f);
  EXPECT_NEAR(0.0f, image[11], 1e-4f);

  EXPECT_NEAR(0.0f, image[12], 1e-4f);
  EXPECT_NEAR(0.0f, image[13], 1e-4f);
  EXPECT_NEAR(0.0f, image[14], 1e-4f);
  EXPECT_NEAR(0.0f, image[15], 1e-4f);
}

