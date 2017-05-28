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


  EXPECT_FLOAT_EQ(0.25382909f, result[0]);
  EXPECT_FLOAT_EQ(0.43275923f, result[1]);
  EXPECT_FLOAT_EQ(0.53250909f, result[2]);
  EXPECT_FLOAT_EQ(0.2457993f, result[3]);
}

TEST(Test_Saliency, SaliencyDownscale)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  std::vector<Eigen::Vector4f> data = { Eigen::Vector4f(0.5f, 0, 1, 1),
                                        Eigen::Vector4f(0.2f, 0, 0, 1),
                                        Eigen::Vector4f(0.1f, 0, -0.5f, 1),
                                        Eigen::Vector4f(0.7f, 0, 1, 1),
                                        Eigen::Vector4f(0.5f, 0, 1, 1),
                                        Eigen::Vector4f(0.2f, 0, 0, 1),
                                        Eigen::Vector4f(0.1f, 0, -0.5f, 1),
                                        Eigen::Vector4f(0.7f, 0, 1, 1),
                                        Eigen::Vector4f(0.7f, 0, 1, 1) };
  auto inputProvider = std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(
      3, 3, data, channelDesc);
  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::Saliency(inputProvider, outputProvider).runKernel();

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());

  EXPECT_FLOAT_EQ(0.57955956f, result[0]);
  EXPECT_FLOAT_EQ(0.58784854f, result[1]);
  EXPECT_FLOAT_EQ(0.70147073f, result[2]);
  EXPECT_FLOAT_EQ(0.27595237f, result[3]);
}


TEST(Test_Saliency, MaxSaliencyLessEqual1)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  std::default_random_engine gen;
  std::uniform_real_distribution<float> dist(0, 1.0f);
  std::vector<Eigen::Vector4f> data(100 * 100);
  for (int i = 0; i < 100 * 100; ++i)
    data[i] = Eigen::Vector4f(dist(gen), dist(gen), dist(gen), 1.0f);

  auto inputProvider = std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(
      100, 100, data, channelDesc);
  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      100, 100, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::Saliency(inputProvider, outputProvider).runKernel();

  auto result = outputProvider->copyDataFromGpu();

  auto maxElement = std::max_element(result.begin(), result.end());
  EXPECT_LE(*maxElement, 1.0f);

}
