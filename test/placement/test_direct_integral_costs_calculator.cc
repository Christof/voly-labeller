#include "../test.h"
#include <Eigen/Core>
#include "../../src/placement/direct_integral_costs_calculator.h"
#include "../cuda_array_mapper.h"
#include "../cuda_array_3d_mapper.h"

TEST(Test_DirectIntegraclCostsCalculator, DirectIntegralCostsCalculator)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  std::vector<Eigen::Vector4f> data = {
    Eigen::Vector4f(0, 0, 0, 0.1f), Eigen::Vector4f(0, 0, 0, 0.7f),
    Eigen::Vector4f(0, 0, 0, 0.4f), Eigen::Vector4f(0, 0, 0, 0.3f),
    Eigen::Vector4f(0, 0, 0, 0.2f), Eigen::Vector4f(0, 0, 0, 0.6f),
    Eigen::Vector4f(0, 0, 0, 0.8f), Eigen::Vector4f(0, 0, 0, 0.9f)
  };
  auto colorProvider = std::make_shared<CudaArray3DMapper<Eigen::Vector4f>>(
      2, 2, 2, data, channelDesc);
  std::vector<float> occlusionData = { 0.1f, 0.7f, 0.4f, 0.3f };

  std::vector<float> saliencyData = { 0.2e5f, 0.1e5f, 0.4e5f, 0.6e5f };
  auto saliencyProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, saliencyData, cudaCreateChannelDesc<float>());

  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::DirectIntegralCostsCalculator calculator(
      colorProvider, saliencyProvider, outputProvider);
  calculator.weights.occlusion = 1.0f;
  calculator.weights.saliency = 1e-3f;
  calculator.runKernel();

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_FLOAT_EQ(1.0f, result[0]);
  EXPECT_FLOAT_EQ(1.0f, result[1]);
  EXPECT_FLOAT_EQ(1.0f, result[2]);
  EXPECT_FLOAT_EQ(1.0f, result[3]);
}
