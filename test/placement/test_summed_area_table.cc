#include "../test.h"

int sumUsingThrustReduce();

TEST(Test_SummedAreaTable, SumUsingThrustReduce)
{
  EXPECT_EQ(6.0f, sumUsingThrustReduce());
}
