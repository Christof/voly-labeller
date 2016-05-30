#include "../test.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include "../../src/placement/distance_transform.h"
#include "../../src/utils/cuda_helper.h"
#include "../cuda_array_mapper.h"

std::vector<float> callDistanceTransform(
    std::shared_ptr<CudaArrayMapper<float>> occlusionImageProvider)
{
  cudaChannelFormatDesc outputChannelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  int pixelCount =
      occlusionImageProvider->getWidth() * occlusionImageProvider->getHeight();
  std::vector<float> resultImage(pixelCount);
  auto output = std::make_shared<CudaArrayMapper<float>>(
      occlusionImageProvider->getWidth(), occlusionImageProvider->getHeight(),
      resultImage, outputChannelDesc);

  Placement::DistanceTransform distanceTransform(occlusionImageProvider,
                                                 output);
  distanceTransform.run();

  resultImage = output->copyDataFromGpu();

  output->unmap();

  return resultImage;
}

TEST(Test_DistanceTransform, DistanceTransform)
{
  std::vector<float> occlusionImage;

  occlusionImage.push_back(0.0f);
  occlusionImage.push_back(0.0f);
  occlusionImage.push_back(0.0f);
  occlusionImage.push_back(0.0f);

  occlusionImage.push_back(0.0f);
  occlusionImage.push_back(1.0f);
  occlusionImage.push_back(1.0f);
  occlusionImage.push_back(0.0f);

  occlusionImage.push_back(0.0f);
  occlusionImage.push_back(1.0f);
  occlusionImage.push_back(1.0f);
  occlusionImage.push_back(0.0f);

  occlusionImage.push_back(0.0f);
  occlusionImage.push_back(0.0f);
  occlusionImage.push_back(0.0f);
  occlusionImage.push_back(0.0f);

  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  auto occlusionImageProvider = std::make_shared<CudaArrayMapper<float>>(
      4, 4, occlusionImage, channelDesc);
  auto image = callDistanceTransform(occlusionImageProvider);

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

