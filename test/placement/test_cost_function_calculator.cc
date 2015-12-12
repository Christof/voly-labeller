#include "../test.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <thrust/device_vector.h>
#include "../cuda_array_mapper.h"
#include "../../src/placement/cost_function_calculator.h"

TEST(Test_CostFunctionCalculator, TestForFirstLabelWithoutConstraints)
{
  const int side = 16;
  std::vector<float> constraintImageValues(side * side, 0.0f);
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  auto constraintImage = std::make_shared<CudaArrayMapper<float>>(
      side, side, constraintImageValues, channelDesc);
  Placement::CostFunctionCalculator calculator(constraintImage);

  calculator.resize(2 * side, 2 * side);
  calculator.setTextureSize(side, side);

  thrust::host_vector<float> occupancy;
  for (int y = 0; y < side; ++y)
  {
    for (int x = 0; x < side; ++x)
    {
      occupancy.push_back(y >= 4 && y < 12 && x >= 4 && x < 12 ? 1.0f : 0.0f);
    }
  }
  thrust::device_vector<float> occupancyDevice = occupancy;

  int labelId = 0;
  int anchorX = 16;
  int anchorY = 12;
  int labelWidthInPixel = 3;
  int labelHeightInPixel = 3;
  auto result =
      calculator.calculateForLabel(occupancyDevice, labelId, anchorX, anchorY,
                                   labelWidthInPixel, labelHeightInPixel);

  EXPECT_EQ(9, std::get<0>(result));
  EXPECT_EQ(6, std::get<1>(result));
}

