#include "../test.h"
#include <Eigen/Core>
#include "../../src/placement/integral_costs_calculator.h"
#include "../cuda_array_mapper.h"

TEST(Test_IntegraclCostsCalculator, IntegralCostsCalculator)
{
  std::vector<float> occlusionData = { 0.1f, 0.7f, 0.4f, 0.3f };
  auto occlusionProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, occlusionData, cudaCreateChannelDesc<float>());

  std::vector<float> saliencyData = { 0.2e5f, 0.1e5f, 0.4e5f, 0.6e5f };
  auto saliencyProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, saliencyData, cudaCreateChannelDesc<float>());
  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::IntegralCostsCalculator calculator(occlusionProvider, saliencyProvider,
                                     outputProvider);
  calculator.weights.occlusion = 1.0f;
  calculator.weights.saliency = 1e-3f;
  calculator.runKernel();

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_FLOAT_EQ(20.1f, result[0]);
  EXPECT_FLOAT_EQ(10.7f, result[1]);
  EXPECT_FLOAT_EQ(40.4f, result[2]);
  EXPECT_FLOAT_EQ(60.3f, result[3]);
}
