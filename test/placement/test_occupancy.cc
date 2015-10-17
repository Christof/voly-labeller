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

  Occupancy(positionsProvider, outputProvider).runKernel();

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_EQ(0.0f, result[0]);
  EXPECT_EQ(1.0f, result[1]);
  EXPECT_EQ(1.0f, result[2]);
  EXPECT_EQ(0.0f, result[3]);
}

