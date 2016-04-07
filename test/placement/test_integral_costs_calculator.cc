#include "../test.h"
#include <Eigen/Core>
#include "../../src/placement/integral_costs_calculator.h"
#include "../cuda_array_mapper.h"

TEST(Test_IntegraclCostsCalculator, IntegralCostsCalculator)
{
  std::vector<float> occlusionData = { 0.1f, 0.7f, 0.4f, 0.3f };
  auto occlusionProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, occlusionData, cudaCreateChannelDesc<float>());
  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::IntegralCostsCalculator(occlusionProvider, outputProvider)
      .runKernel();

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_EQ(0.1f, result[0]);
  EXPECT_EQ(0.7f, result[1]);
  EXPECT_EQ(0.4f, result[2]);
  EXPECT_EQ(0.3f, result[3]);
}
