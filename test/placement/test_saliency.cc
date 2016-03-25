#include "../test.h"
#include <Eigen/Core>
#include <thrust/host_vector.h>
#include "../../src/placement/saliency.h"
#include "../cuda_array_mapper.h"

TEST(Test_Saliency, Saliency)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  std::vector<Eigen::Vector4f> data = { Eigen::Vector4f(0.5f, 0, 1, 1),
                                        Eigen::Vector4f(0.2f, 0, 0, 1),
                                        Eigen::Vector4f(0.1f, 0, -0.5f, 1),
                                        Eigen::Vector4f(0.7f, 0, 1, 1) };
  auto inputProvider = std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(
      2, 2, data, channelDesc);
  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::Saliency(inputProvider, outputProvider).runKernel();

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  /*
  EXPECT_EQ(1.1f, result[0]);
  EXPECT_EQ(-1.1f, result[1]);
  EXPECT_EQ(1.6f, result[2]);
  EXPECT_EQ(-0.7f, result[3]);
  */

  EXPECT_EQ(0.9f, result[0]);
  EXPECT_EQ(1.5f, result[1]);
  EXPECT_EQ(-1.2f, result[2]);
  EXPECT_EQ(-0.9f, result[3]);
}
