#include "./test.h"
#include "../src/volume_reader.h"

TEST(IntegrationTest_VolumeReader, test)
{
  VolumeReader reader("assets/datasets/MANIX.mhd");

  auto size = reader.getSize();
  EXPECT_EQ(512, size.x());
  EXPECT_EQ(512, size.y());
  EXPECT_EQ(460, size.z());

  EXPECT_TRUE(reader.isCT());

  auto matrix = reader.getTransformationMatrix();
  Eigen::Matrix4f expected;
  expected << 1, 0, 0, -125, 0, 1, 0, -63, 0, 0, 1, -858, 0, 0, 0, 1;
  EXPECT_Matrix4f_NEAR(expected, matrix, 1E-5);
}
