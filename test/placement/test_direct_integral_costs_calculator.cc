#include "../test.h"
#include <Eigen/Core>
#include "../../src/placement/direct_integral_costs_calculator.h"
#include "../cuda_array_mapper.h"
#include "../cuda_array_3d_mapper.h"

TEST(Test_DirectIntegraclCostsCalculator,
     DirectIntegralCostsCalculatorForFirstLayer)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  std::vector<Eigen::Vector4f> data = {
    Eigen::Vector4f(0, 0, 0, 0.1f), Eigen::Vector4f(0, 0, 0, 0.7f),
    Eigen::Vector4f(0, 0, 0, 0.4f), Eigen::Vector4f(0, 0, 0, 0.3f),
    Eigen::Vector4f(0, 0, 0, 0.2f), Eigen::Vector4f(0, 0, 0, 0.6f),
    Eigen::Vector4f(0, 0, 0, 0.8f), Eigen::Vector4f(0, 0, 0, 0.9f),
    Eigen::Vector4f(0, 0, 0, 0.1f), Eigen::Vector4f(0, 0, 0, 0.8f),
    Eigen::Vector4f(0, 0, 0, 0.4f), Eigen::Vector4f(0, 0, 0, 0.2f)
  };
  int layerCount = 3;
  auto colorProvider = std::make_shared<CudaArray3DMapper<Eigen::Vector4f>>(
      2, 2, layerCount, data, channelDesc);

  std::vector<float> saliencyData = { 0.5f, 0.1f, 0.4f, 0.6f };
  auto saliencyProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, saliencyData, cudaCreateChannelDesc<float>());

  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::DirectIntegralCostsCalculator calculator(
      colorProvider, saliencyProvider, outputProvider);
  calculator.weights.occlusion = 1.0f;
  calculator.weights.saliency = 1.0f;
  calculator.weights.fixOcclusionPart = 0.2f;
  int layerIndex = 0;
  calculator.runKernel(layerIndex, layerCount);

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_FLOAT_EQ((0.1f + (0.2f + 0.8f * 0.5f) * (0.2f + 0.1f)) / layerCount,
                  result[0]);
  EXPECT_FLOAT_EQ((0.7f + (0.2f + 0.8f * 0.1f) * (0.6f + 0.8f)) / layerCount,
                  result[1]);
  EXPECT_FLOAT_EQ((0.4f + (0.2f + 0.8f * 0.4f) * (0.8f + 0.4f)) / layerCount,
                  result[2]);
  EXPECT_FLOAT_EQ((0.3f + (0.2f + 0.8f * 0.6f) * (0.9f + 0.2f)) / layerCount,
                  result[3]);
}

TEST(Test_DirectIntegraclCostsCalculator,
     DirectIntegralCostsCalculatorForMiddleLayer)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  std::vector<Eigen::Vector4f> data = {
    Eigen::Vector4f(0, 0, 0, 0.1f), Eigen::Vector4f(0, 0, 0, 0.7f),
    Eigen::Vector4f(0, 0, 0, 0.4f), Eigen::Vector4f(0, 0, 0, 0.3f),
    Eigen::Vector4f(0, 0, 0, 0.2f), Eigen::Vector4f(0, 0, 0, 0.6f),
    Eigen::Vector4f(0, 0, 0, 0.8f), Eigen::Vector4f(0, 0, 0, 0.9f),
    Eigen::Vector4f(0, 0, 0, 0.1f), Eigen::Vector4f(0, 0, 0, 0.8f),
    Eigen::Vector4f(0, 0, 0, 0.4f), Eigen::Vector4f(0, 0, 0, 0.2f)
  };
  int layerCount = 3;
  auto colorProvider = std::make_shared<CudaArray3DMapper<Eigen::Vector4f>>(
      2, 2, layerCount, data, channelDesc);

  std::vector<float> saliencyData = { 0.5f, 0.1f, 0.4f, 0.6f };
  auto saliencyProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, saliencyData, cudaCreateChannelDesc<float>());

  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::DirectIntegralCostsCalculator calculator(
      colorProvider, saliencyProvider, outputProvider);
  calculator.weights.occlusion = 1.0f;
  calculator.weights.saliency = 1.0f;
  calculator.weights.fixOcclusionPart = 0.2f;
  int layerIndex = 1;
  calculator.runKernel(layerIndex, layerCount);

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_FLOAT_EQ((0.1f + 0.2f + (0.2f + 0.8f * 0.5f) * 0.1f) / layerCount,
                  result[0]);
  EXPECT_FLOAT_EQ((0.7f + 0.6f + (0.2f + 0.8f * 0.1f) * 0.8f) / layerCount,
                  result[1]);
  EXPECT_FLOAT_EQ((0.4f + 0.8f + (0.2f + 0.8f * 0.4f) * 0.4f) / layerCount,
                  result[2]);
  EXPECT_FLOAT_EQ((0.3f + 0.9f + (0.2f + 0.8f * 0.6f) * 0.2f) / layerCount,
                  result[3]);
}

TEST(Test_DirectIntegraclCostsCalculator,
     DirectIntegralCostsCalculatorForLastLayer)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  std::vector<Eigen::Vector4f> data = {
    Eigen::Vector4f(0, 0, 0, 0.1f), Eigen::Vector4f(0, 0, 0, 0.7f),
    Eigen::Vector4f(0, 0, 0, 0.4f), Eigen::Vector4f(0, 0, 0, 0.3f),
    Eigen::Vector4f(0, 0, 0, 0.2f), Eigen::Vector4f(0, 0, 0, 0.6f),
    Eigen::Vector4f(0, 0, 0, 0.8f), Eigen::Vector4f(0, 0, 0, 0.9f),
    Eigen::Vector4f(0, 0, 0, 0.1f), Eigen::Vector4f(0, 0, 0, 0.8f),
    Eigen::Vector4f(0, 0, 0, 0.4f), Eigen::Vector4f(0, 0, 0, 0.2f)
  };
  int layerCount = 3;
  auto colorProvider = std::make_shared<CudaArray3DMapper<Eigen::Vector4f>>(
      2, 2, layerCount, data, channelDesc);

  std::vector<float> saliencyData = { 0.5f, 0.1f, 0.4f, 0.6f };
  auto saliencyProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, saliencyData, cudaCreateChannelDesc<float>());

  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::DirectIntegralCostsCalculator calculator(
      colorProvider, saliencyProvider, outputProvider);
  calculator.weights.occlusion = 1.0f;
  calculator.weights.saliency = 1.0f;
  calculator.weights.fixOcclusionPart = 0.2f;
  int layerIndex = 2;
  calculator.runKernel(layerIndex, layerCount);

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_FLOAT_EQ((0.1f + 0.2f + 0.1f) / layerCount, result[0]);
  EXPECT_FLOAT_EQ((0.7f + 0.6f + 0.8f) / layerCount, result[1]);
  EXPECT_FLOAT_EQ((0.4f + 0.8f + 0.4f) / layerCount, result[2]);
  EXPECT_FLOAT_EQ((0.3f + 0.9f + 0.2f) / layerCount, result[3]);
}

TEST(Test_DirectIntegraclCostsCalculator,
     DirectIntegralCostsCalculatorForLastLayerDownscaling)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  std::vector<Eigen::Vector4f> data = {
    Eigen::Vector4f(0, 0, 0, 0.1f), Eigen::Vector4f(0, 0, 0, 0.7f),
    Eigen::Vector4f(0, 0, 0, 0.4f), Eigen::Vector4f(0, 0, 0, 0.3f),
    Eigen::Vector4f(0, 0, 0, 0.2f), Eigen::Vector4f(0, 0, 0, 0.6f),
    Eigen::Vector4f(0, 0, 0, 0.8f), Eigen::Vector4f(0, 0, 0, 0.9f),
    Eigen::Vector4f(0, 0, 0, 0.1f), Eigen::Vector4f(0, 0, 0, 0.8f),
    Eigen::Vector4f(0, 0, 0, 0.4f), Eigen::Vector4f(0, 0, 0, 0.2f),
    Eigen::Vector4f(0, 0, 0, 0.8f), Eigen::Vector4f(0, 0, 0, 0.9f),
    Eigen::Vector4f(0, 0, 0, 0.1f), Eigen::Vector4f(0, 0, 0, 0.8f),
    Eigen::Vector4f(0, 0, 0, 0.4f), Eigen::Vector4f(0, 0, 0, 0.2f)
  };
  int layerCount = 2;
  auto colorProvider = std::make_shared<CudaArray3DMapper<Eigen::Vector4f>>(
      3, 3, layerCount, data, channelDesc);

  std::vector<float> saliencyData = { 0.5f, 0.1f, 0.4f, 0.6f };
  auto saliencyProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, saliencyData, cudaCreateChannelDesc<float>());

  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::DirectIntegralCostsCalculator calculator(
      colorProvider, saliencyProvider, outputProvider);
  calculator.weights.occlusion = 1.0f;
  calculator.weights.saliency = 1.0f;
  calculator.weights.fixOcclusionPart = 0.2f;
  int layerIndex = 1;
  calculator.runKernel(layerIndex, layerCount);

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_FLOAT_EQ(0.45f, result[0]);
  EXPECT_FLOAT_EQ(0.425f, result[1]);
  EXPECT_FLOAT_EQ(0.675f, result[2]);
  EXPECT_FLOAT_EQ(0.425f, result[3]);
}


TEST(Test_DirectIntegraclCostsCalculator,
     DirectIntegralCostsCalculatorForSingleLayerReturnsWeightedSum)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  std::vector<Eigen::Vector4f> data = {
    Eigen::Vector4f(0, 0, 0, 0.1f), Eigen::Vector4f(0, 0, 0, 0.7f),
    Eigen::Vector4f(0, 0, 0, 0.4f), Eigen::Vector4f(0, 0, 0, 0.3f),
  };
  int layerCount = 1;
  auto colorProvider = std::make_shared<CudaArray3DMapper<Eigen::Vector4f>>(
      2, 2, layerCount, data, channelDesc);

  std::vector<float> saliencyData = { 0.5f, 0.1f, 0.4f, 0.6f };
  auto saliencyProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, saliencyData, cudaCreateChannelDesc<float>());

  auto outputProvider = std::make_shared<CudaArrayMapper<float>>(
      2, 2, std::vector<float>(4), cudaCreateChannelDesc<float>());

  Placement::DirectIntegralCostsCalculator calculator(
      colorProvider, saliencyProvider, outputProvider);
  calculator.weights.occlusion = 2.0f;
  calculator.weights.saliency = 3.0f;
  int layerIndex = 0;
  calculator.runKernel(layerIndex, layerCount);

  auto result = outputProvider->copyDataFromGpu();

  ASSERT_EQ(4, result.size());
  EXPECT_FLOAT_EQ((2 * 0.1f + 3 * 0.5f), result[0]);
  EXPECT_FLOAT_EQ((2 * 0.7f + 3 * 0.1f), result[1]);
  EXPECT_FLOAT_EQ((2 * 0.4f + 3 * 0.4f), result[2]);
  EXPECT_FLOAT_EQ((2 * 0.3f + 3 * 0.6f), result[3]);
}
