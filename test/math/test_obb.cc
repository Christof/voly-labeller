#include "../test.h"
#include "../../src/math/obb.h"

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
