#include "../test.h"
#include <thrust/host_vector.h>
#include "../cuda_array_mapper.h"
#include "../../src/placement/summed_area_table.h"

TEST(Test_SummedAreaTable, SATWithoutClass)
{
  std::vector<float> input = { 1, 2, 3, 4 };
  thrust::host_vector<float> result = algSAT(input.data(), 2, 2);

  ASSERT_LE(4, result.size());

  EXPECT_EQ(1, result[0]);
  EXPECT_EQ(3, result[1]);
  EXPECT_EQ(4, result[2]);
  EXPECT_EQ(10, result[3]);
}

TEST(Test_SummedAreaTable, SAT)
{
  std::vector<float> input = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  auto inputImageProvider =
      std::make_shared<CudaArrayMapper<float>>(4, 4, input, channelDesc);

  SummedAreaTable table(inputImageProvider);
  table.runKernel();
  thrust::host_vector<float> result = table.getResults();

  ASSERT_LE(16, result.size());

  EXPECT_EQ(1, result[0]);
  EXPECT_EQ(3, result[1]);
  EXPECT_EQ(6, result[2]);
  EXPECT_EQ(10, result[3]);
  EXPECT_EQ(6, result[4]);
  EXPECT_EQ(14, result[5]);
  EXPECT_EQ(24, result[6]);
  EXPECT_EQ(36, result[7]);
}
