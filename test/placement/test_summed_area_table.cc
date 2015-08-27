#include "../test.h"
#include <thrust/host_vector.h>

int sumUsingThrustReduce();
int sumUsingCuda();
int sumUsingCudaInLib();
thrust::host_vector<float> algSAT(float *h_inout, int w, int h);

TEST(Test_SummedAreaTable, SumUsingThrustReduce)
{
  EXPECT_EQ(6.0f, sumUsingThrustReduce());
}

TEST(Test_SummedAreaTable, SumUsingCuda)
{
  EXPECT_EQ(6.0f, sumUsingCuda());
}

TEST(Test_SummedAreaTable, SumUsingCudaInLib)
{
  EXPECT_EQ(6.0f, sumUsingCudaInLib());
}

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
