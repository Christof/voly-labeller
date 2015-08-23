#include "../test.h"

int sumUsingThrustReduce();
int sumUsingCuda();
int sumUsingCudaInLib();

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
