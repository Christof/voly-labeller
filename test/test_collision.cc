#include "./test.h"
#include "../src/collision.h"

TEST(Test_test2DSegmentSegment, Intersection)
{
  EXPECT_TRUE(
      test2DSegmentSegment(Eigen::Vector2f(-1, -1), Eigen::Vector2f(1, 1),
                           Eigen::Vector2f(-1, 1), Eigen::Vector2f(1, -1)));
}

TEST(Test_test2DSegmentSegment, NoIntersectionForParallelSegments)
{
  EXPECT_FALSE(
      test2DSegmentSegment(Eigen::Vector2f(0, 0), Eigen::Vector2f(1, 0),
                           Eigen::Vector2f(0, 1), Eigen::Vector2f(1, 1)));
}

TEST(Test_test2DSegmentSegment, NoIntersection)
{
  EXPECT_FALSE(
      test2DSegmentSegment(Eigen::Vector2f(-1, -1), Eigen::Vector2f(1, 1),
                           Eigen::Vector2f(-1, 1), Eigen::Vector2f(-0.5f, 0.5f)));
}
