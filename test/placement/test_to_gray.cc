#include "../test.h"
#include "../../src/placement/to_gray.h"
#include "../cuda_array_mapper.h"
#include <thrust/host_vector.h>

unsigned int toGrayUsingCuda(unsigned int value)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
  std::vector<unsigned int> data = { value };
  auto arrayProvider =
      std::make_shared<CudaArrayMapper<unsigned int>>(1, 1, data, channelDesc);

  ToGray(arrayProvider).runKernel();

  auto resultVector = arrayProvider->copyDataFromGpu();

  return resultVector[0];
}

TEST(Test_ToGray, toGray)
{
  unsigned int input = 0xFF00007F;
  auto result = toGrayUsingCuda(input);
  EXPECT_EQ(0xFF252525, result);
}

