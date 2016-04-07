#include "../test.h"
#include <Eigen/Core>
#include <thrust/host_vector.h>
#include "../../src/placement/occlusion.h"
#include "../cuda_array_mapper.h"

TEST(Test_Occlusion, Occlusion)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  std::vector<Eigen::Vector4f> data = { Eigen::Vector4f(0, 0, 0, 0.1f),
                                        Eigen::Vector4f(0, 0, 0, 0.7f),
                                        Eigen::Vector4f(0, 0, 0, 0.4f),
                                        Eigen::Vector4f(0, 0, 0, 0.3f) };
  auto colorProvider = std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(
      2, 2, data, channelDesc);
  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  std::vector<std::shared_ptr<CudaArrayProvider>> colorProviders;
  colorProviders.push_back(colorProvider);
  Placement::Occlusion(colorProviders, outputProvider).calculateOcclusion();

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_EQ(0.1f, result[0]);
  EXPECT_EQ(0.7f, result[1]);
  EXPECT_EQ(0.4f, result[2]);
  EXPECT_EQ(0.3f, result[3]);
}

TEST(Test_Occlusion, OccupancyWithSamplingShouldUseMaxAlphaValue)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  /*
   *  0.4    0  0.5  0.4
   *  0.5  0.6  0.7  0.5
   *  0.7  0.1  0.3  0.2
   *    1    0  0.5  0.9
   */
  std::vector<Eigen::Vector4f> data = {
    Eigen::Vector4f(0, 0, 0, 0.4f), Eigen::Vector4f(0, 0, 0, 0),
    Eigen::Vector4f(0, 0, 0, 0.5f), Eigen::Vector4f(0, 0, 0, 0.4f),

    Eigen::Vector4f(0, 0, 0, 0.5f), Eigen::Vector4f(0, 0, 0, 0.6f),
    Eigen::Vector4f(0, 0, 0, 0.7f), Eigen::Vector4f(0, 0, 0, 0.5f),

    Eigen::Vector4f(0, 0, 0, 0.7f), Eigen::Vector4f(0, 0, 0, 0.1),
    Eigen::Vector4f(0, 0, 0, 0.3f), Eigen::Vector4f(0, 0, 0, 0.2f),

    Eigen::Vector4f(0, 0, 0, 1),    Eigen::Vector4f(0, 0, 0, 0),
    Eigen::Vector4f(0, 0, 0, 0.5f), Eigen::Vector4f(0, 0, 0, 0.9)
  };
  auto colorProvider = std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(
      4, 4, data, channelDesc);
  std::vector<std::shared_ptr<CudaArrayProvider>> colorProviders;
  colorProviders.push_back(colorProvider);
  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::Occlusion(colorProviders, outputProvider).calculateOcclusion();

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_EQ(0.6f, result[0]);
  EXPECT_EQ(0.7f, result[1]);
  EXPECT_EQ(1.0f, result[2]);
  EXPECT_EQ(0.9f, result[3]);
}

TEST(Test_Occlusion, AddOcclusion)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  std::vector<Eigen::Vector4f> data = { Eigen::Vector4f(0, 0, 0, 0.1f),
                                        Eigen::Vector4f(0, 0, 0, 0.7f),
                                        Eigen::Vector4f(0, 0, 0, 0.4f),
                                        Eigen::Vector4f(0, 0, 0, 0.3f) };
  auto colorProvider = std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(
      2, 2, data, channelDesc);
  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  std::vector<std::shared_ptr<CudaArrayProvider>> colorProviders;
  colorProviders.push_back(colorProvider);

  Placement::Occlusion occlusion(colorProviders, outputProvider);
  occlusion.calculateOcclusion();
  occlusion.addOcclusion();

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_EQ(0.2f, result[0]);
  EXPECT_EQ(1.4f, result[1]);
  EXPECT_EQ(0.8f, result[2]);
  EXPECT_EQ(0.6f, result[3]);
}

