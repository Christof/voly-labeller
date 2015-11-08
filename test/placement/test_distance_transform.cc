#include "../test.h"
#include <cuda_runtime_api.h>
#include <memory>
#include "../cuda_array_mapper.h"

std::vector<float> callDistanceTransform(
    std::shared_ptr<CudaArrayMapper<float>> depthImageProvider,
    std::vector<float> &resultVector);

TEST(Test_DistanceTransform, DistanceTransform)
{
  std::vector<float> depthImage;

  depthImage.push_back(1.0f);
  depthImage.push_back(1.0f);
  depthImage.push_back(1.0f);
  depthImage.push_back(1.0f);

  depthImage.push_back(1.0f);
  depthImage.push_back(0.0f);
  depthImage.push_back(0.0f);
  depthImage.push_back(1.0f);

  depthImage.push_back(1.0f);
  depthImage.push_back(0.0f);
  depthImage.push_back(0.0f);
  depthImage.push_back(1.0f);

  depthImage.push_back(1.0f);
  depthImage.push_back(1.0f);
  depthImage.push_back(1.0f);
  depthImage.push_back(1.0f);

  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  auto depthImageProvider =
      std::make_shared<CudaArrayMapper<float>>(4, 4, depthImage, channelDesc);
  std::vector<float> resultVector;
  auto image = callDistanceTransform(depthImageProvider, resultVector);

  EXPECT_EQ(16, resultVector.size());
  EXPECT_NEAR(0.0f, resultVector[0], 1e-4f);
  EXPECT_NEAR(0.0f, resultVector[1], 1e-4f);
  EXPECT_NEAR(0.0f, resultVector[2], 1e-4f);
  EXPECT_NEAR(0.0f, resultVector[3], 1e-4f);

  EXPECT_NEAR(0.0f, resultVector[4], 1e-4f);
  EXPECT_NEAR(1.0f / 32, resultVector[5], 1e-4f);
  EXPECT_NEAR(1.0f / 32, resultVector[6], 1e-4f);
  EXPECT_NEAR(0.0f, resultVector[7], 1e-4f);

  EXPECT_NEAR(0.0f, resultVector[8], 1e-4f);
  EXPECT_NEAR(1.0f / 32, resultVector[9], 1e-4f);
  EXPECT_NEAR(1.0f / 32, resultVector[10], 1e-4f);
  EXPECT_NEAR(0.0f, resultVector[11], 1e-4f);

  EXPECT_NEAR(0.0f, resultVector[12], 1e-4f);
  EXPECT_NEAR(0.0f, resultVector[13], 1e-4f);
  EXPECT_NEAR(0.0f, resultVector[14], 1e-4f);
  EXPECT_NEAR(0.0f, resultVector[15], 1e-4f);

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

