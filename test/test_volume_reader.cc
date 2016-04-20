#include "./test.h"
#include "../src/volume_reader.h"

TEST(IntegrationTest_VolumeReader, TestReadingOfCTData)
{
  VolumeReader reader("assets/datasets/MANIX.mhd");

  auto size = reader.getSize();
  EXPECT_EQ(512, size.x());
  EXPECT_EQ(512, size.y());
  EXPECT_EQ(460, size.z());

  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.0004882f, 0.0004882f, 0.0007f),
                       reader.getSpacing(), 1E-7);

  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.25f, 0.25f, 0.322f),
                       reader.getPhysicalSize(), 1E-4);

  auto transformation = reader.getTransformationMatrix();
  Eigen::Matrix4f expectedTransformation;
  expectedTransformation << 1, 0, 0, 0, 0, 1, 0, 0.062f, 0, 0, 1, -0.69699f, 0, 0, 0,
      1;
  EXPECT_Matrix4f_NEAR(expectedTransformation, transformation, 1E-5);
}

