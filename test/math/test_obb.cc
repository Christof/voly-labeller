#include "../test.h"
#include <Eigen/Geometry>
#include "../../src/math/obb.h"

TEST(Test_Obb, CreationWithDefaultConstructor)
{
  Math::Obb obb;

  EXPECT_FALSE(obb.isInitialized());
}

TEST(Test_Obb, CreationFromPoints)
{
  Eigen::MatrixXf points = Eigen::MatrixXf::Zero(3, 8);
  points.col(0) = Eigen::Vector3f(-1, -1, -1);
  points.col(1) = Eigen::Vector3f(-1, 1, 1);
  points.col(2) = Eigen::Vector3f(1, -1, 1);
  points.col(3) = Eigen::Vector3f(1, 1, -1);
  points.col(4) = Eigen::Vector3f(1, 1, 1);
  points.col(5) = Eigen::Vector3f(-1, -1, 1);
  points.col(6) = Eigen::Vector3f(1, -1, -1);
  points.col(7) = Eigen::Vector3f(-1, 1, -1);

  Math::Obb obb(points);

  EXPECT_TRUE(obb.isInitialized());

  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0, 0, 0), obb.getCenter(), 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1, 1, 1), obb.getHalfWidths(), 1E-4);

  EXPECT_Vector3f_NEAR(Eigen::Vector3f(-1, -1, -1), obb.corners[0], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(-1, -1, 1), obb.corners[1], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1, -1, 1), obb.corners[2], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1, -1, -1), obb.corners[3], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(-1, 1, -1), obb.corners[4], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(-1, 1, 1), obb.corners[5], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1, 1, 1), obb.corners[6], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1, 1, -1), obb.corners[7], 1E-4);
}

TEST(Test_Obb, CreationFromCenterHalfWidthsAndAxes)
{
  Eigen::Vector3f center(1, 2, 3);
  Eigen::Vector3f halfWidths(0.1f, 0.2f, 0.3f);
  Eigen::Matrix3f axes;
  axes << 0, 0, 1, 0, 1, 0, 1, 0, 0;
  Math::Obb obb(center, halfWidths, axes);

  EXPECT_TRUE(obb.isInitialized());

  EXPECT_Vector3f_NEAR(center, obb.getCenter(), 1E-4);
  EXPECT_Vector3f_NEAR(halfWidths, obb.getHalfWidths(), 1E-4);

  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.7f, 1.8f, 2.9f), obb.corners[0], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.7f, 1.8f, 3.1f), obb.corners[1], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1.3f, 1.8f, 3.1f), obb.corners[2], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1.3f, 1.8f, 2.9f), obb.corners[3], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.7f, 2.2f, 2.9f), obb.corners[4], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.7f, 2.2f, 3.1f), obb.corners[5], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1.3f, 2.2f, 3.1f), obb.corners[6], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1.3f, 2.2f, 2.9f), obb.corners[7], 1E-4);
}

TEST(Test_Obb, ApplyTransformationMatrix)
{
  Eigen::Vector3f center(0, 0, 0);
  Eigen::Vector3f halfWidths(0.1f, 0.2f, 0.3f);
  Eigen::Matrix3f axes;
  axes << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  Math::Obb untransformedObb(center, halfWidths, axes);

  Eigen::Affine3f transformation(
      Eigen::Translation3f(Eigen::Vector3f(1, 2, 3)) *
      Eigen::AngleAxisf(0.5 * M_PI, Eigen::Vector3f::UnitY()));
  Eigen::Matrix4f matrix = transformation.matrix();

  Math::Obb obb = untransformedObb * matrix;

  EXPECT_TRUE(obb.isInitialized());

  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1, 2, 3), obb.getCenter(), 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.3f, 0.2f, -0.1f), obb.getHalfWidths(), 1E-4);

  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.7f, 1.8f, 3.1f), obb.corners[0], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.7f, 1.8f, 2.9f), obb.corners[1], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1.3f, 1.8f, 2.9f), obb.corners[2], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1.3f, 1.8f, 3.1f), obb.corners[3], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.7f, 2.2f, 3.1f), obb.corners[4], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.7f, 2.2f, 2.9f), obb.corners[5], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1.3f, 2.2f, 2.9f), obb.corners[6], 1E-4);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1.3f, 2.2f, 3.1f), obb.corners[7], 1E-4);
}

