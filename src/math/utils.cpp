#include "./utils.h"

namespace Math
{

int computeNextPowerOfTwo(int value)
{
  int powerOfTwo = 1;
  while (powerOfTwo < value)
    powerOfTwo <<= 1;

  return powerOfTwo;
}

}  // namespace Math
