#include "../test.h"
#include "../../src/math/aabb2d.h"

TEST(Test_Aabb2d, testAabbReturnsTrueIfBoundingBoxesOverlap)
{
  Math::Aabb2d a(Eigen::Vector2f(0, 0), Eigen::Vector2f(1, 1));
  Math::Aabb2d b(Eigen::Vector2f(0.5f, 0.5f), Eigen::Vector2f(1, 1));

  EXPECT_TRUE(a.testAabb(b));
}

TEST(Test_Aabb2d, testAabbReturnsFalseIfBoundingBoxesDontOverlap)
{
  Math::Aabb2d a(Eigen::Vector2f(0, 0), Eigen::Vector2f(1, 1));
  Math::Aabb2d b(Eigen::Vector2f(3, 0), Eigen::Vector2f(1, 1));

  EXPECT_FALSE(a.testAabb(b));
}

TEST(Test_Aabb2d, testAabbReturnsFalseIfBoundingBoxesDontOverlapOnOtherAxis)
{
  Math::Aabb2d a(Eigen::Vector2f(0, 0), Eigen::Vector2f(1, 1));
  Math::Aabb2d b(Eigen::Vector2f(0, 3), Eigen::Vector2f(1, 1));

  EXPECT_FALSE(a.testAabb(b));
}
