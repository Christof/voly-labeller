#include "./test.h"
#include "../src/volume_reader.h"

TEST(Test_VolumeReader, test)
{
  VolumeReader reader("assets/datasets/MANIX.mhd");

  auto size = reader.getSize();
  EXPECT_EQ(512, size.x());
  EXPECT_EQ(512, size.y());
  EXPECT_EQ(460, size.z());
}
