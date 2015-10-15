#include "../test.h"
#include <thrust/host_vector.h>

unsigned int toGrayUsingCuda(unsigned int value);

TEST(Test_ToGray, toGray)
{
  unsigned int input = 0xFF00007F;
  auto result = toGrayUsingCuda(input);
  EXPECT_EQ(0xFF252525, result);
}

