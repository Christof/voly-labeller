#include "../test.h"
#include <thrust/host_vector.h>
#include "../cuda_array_mapper.h"
#include "../../src/placement/summed_area_table.h"

TEST(Test_SummedAreaTable, SummedAreaTable)
{
  std::vector<float> input(32 * 32, 0);
  for (int i = 0; i < 4; ++i)
  {
    input[i] = i + 1;
    input[i + 32] = i + 5;
    input[i + 64] = i + 1;
    input[i + 96] = i + 5;
  }
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  auto inputImageProvider =
      std::make_shared<CudaArrayMapper<float>>(32, 32, input, channelDesc);

  SummedAreaTable table(inputImageProvider);
  table.runKernel();
  thrust::host_vector<float> result = table.getResults();

  ASSERT_LE(32 * 32, result.size());

  std::cout << "result: " << std::endl;
  printVectorAsMatrix(result, 32, 32);

  EXPECT_EQ(1, result[0]);
  EXPECT_EQ(3, result[1]);
  EXPECT_EQ(6, result[2]);
  EXPECT_EQ(10, result[3]);
  EXPECT_EQ(10, result[4]);
  EXPECT_EQ(10, result[31]);
  EXPECT_EQ(6, result[32]);
  EXPECT_EQ(14, result[33]);
  EXPECT_EQ(24, result[34]);
  EXPECT_EQ(36, result[35]);
}

