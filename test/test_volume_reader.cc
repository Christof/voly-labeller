#include "./test.h"
#include "../src/volume_reader.h"

TEST(IntegrationTest_VolumeReader, TestReadingOfCTData)
{
  VolumeReader reader("assets/datasets/MANIX.mhd");

  auto size = reader.getSize();
  EXPECT_EQ(512, size.x());
  EXPECT_EQ(512, size.y());
  EXPECT_EQ(460, size.z());

  EXPECT_TRUE(reader.isCT());

  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.4882f, 0.4882f, 0.7f),
                       reader.getSpacing(), 1E-4);

  auto transformation = reader.getTransformationMatrix();
  Eigen::Matrix4f expectedTransformation;
  expectedTransformation << 1, 0, 0, -125, 0, 1, 0, -63, 0, 0, 1, -858, 0, 0, 0,
      1;
  EXPECT_Matrix4f_NEAR(expectedTransformation, transformation, 1E-5);
}

