#include "../test.h"
#include <cuda_runtime_api.h>
#include <memory>

float calculateCosts();

TEST(Test_CostFunctionCalculator, TryFunctionPointsComposition)
{
  EXPECT_EQ(3.0f, calculateCosts());
}
