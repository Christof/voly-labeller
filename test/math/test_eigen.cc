#include "../test.h"
#include "../../src/math/eigen.h"

TEST(Test_Eigen, toVector3f)
{
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(1, 2, 3),
                       toVector3f(Eigen::Vector4f(1, 2, 3, 4)), 1E-6);
}

TEST(Test_Eigen, toVector4f)
{
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 2, 3, 1),
                       toVector4f(Eigen::Vector3f(1, 2, 3)), 1E-6);
}

TEST(Test_Eigen, mul)
{
  Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
  matrix(0, 0) = 2;
  matrix(1, 1) = 3;
  matrix(2, 2) = 4;
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(2, 6, 12, 1),
                       mul(matrix, Eigen::Vector3f(1, 2, 3)), 1E-6);
}

TEST(Test_Eigen, toEigenForQPoint)
{
  EXPECT_Vector2f_NEAR(Eigen::Vector2f(1, 2), toEigen(QPoint(1, 2)), 1E-6);
}

TEST(Test_Eigen, toEigenForQPointF)
{
  EXPECT_Vector2f_NEAR(Eigen::Vector2f(1.1f, 2.2f),
                       toEigen(QPointF(1.1f, 2.2f)), 1E-6);
}
