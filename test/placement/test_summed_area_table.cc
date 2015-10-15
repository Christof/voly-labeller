#include "../test.h"
#include <thrust/host_vector.h>
#include "../../src/placement/summed_area_table.h"

TEST(Test_SummedAreaTable, SAT)
{
  std::vector<float> input = { 1, 2, 3, 4 };
  thrust::host_vector<float> result = algSAT(input.data(), 2, 2);

  ASSERT_LE(4, result.size());

  EXPECT_EQ(1, result[0]);
  EXPECT_EQ(3, result[1]);
  EXPECT_EQ(4, result[2]);
  EXPECT_EQ(10, result[3]);
}

