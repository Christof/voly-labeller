#include "../test.h"
#include <Eigen/Core>
#include <thrust/host_vector.h>
#include "../../src/placement/occupancy.h"
#include "../cuda_array_mapper.h"

TEST(Test_Occupancy, Occupancy)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  std::vector<Eigen::Vector4f> data = { Eigen::Vector4f(0, 0, 1, 1),
                                        Eigen::Vector4f(0, 0, 0, 1),
                                        Eigen::Vector4f(0, 0, -0.5f, 1),
                                        Eigen::Vector4f(0, 0, 1, 1) };
  auto positionsProvider = std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(
      2, 2, data, channelDesc);
  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::Occupancy(positionsProvider, outputProvider).runKernel();

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_EQ(0.0f, result[0]);
  EXPECT_EQ(1.0f, result[1]);
  EXPECT_EQ(1.5f, result[2]);
  EXPECT_EQ(0.0f, result[3]);
}

TEST(Test_Occupancy, OccupancyWithSamplingShouldUseMaxDepthValue)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  /*
   *    1    0 -0.5  0.4
   *  0.5    0 -0.5 -0.5
   *  0.7    0 -0.5  0.2
   *   -1    0 -0.5   -1
   */
  std::vector<Eigen::Vector4f> data = {
    Eigen::Vector4f(0, 0, 1, 1),     Eigen::Vector4f(0, 0, 0, 1),
    Eigen::Vector4f(0, 0, -0.5f, 1), Eigen::Vector4f(0, 0, 0.4f, 1),

    Eigen::Vector4f(0, 0, 0.5f, 1),  Eigen::Vector4f(0, 0, 0, 1),
    Eigen::Vector4f(0, 0, -0.5f, 1), Eigen::Vector4f(0, 0, -0.5f, 1),

    Eigen::Vector4f(0, 0, 0.7f, 1),  Eigen::Vector4f(0, 0, 0, 1),
    Eigen::Vector4f(0, 0, -0.5f, 1), Eigen::Vector4f(0, 0, 0.2f, 1),

    Eigen::Vector4f(0, 0, -1, 1),    Eigen::Vector4f(0, 0, 0, 1),
    Eigen::Vector4f(0, 0, -0.5f, 1), Eigen::Vector4f(0, 0, -1, 1)
  };
  auto positionsProvider = std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(
      4, 4, data, channelDesc);
  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::Occupancy(positionsProvider, outputProvider).runKernel();

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_EQ(0.0f, result[0]);
  EXPECT_EQ(0.6f, result[1]);
  EXPECT_EQ(0.3f, result[2]);
  EXPECT_EQ(0.8f, result[3]);
}
