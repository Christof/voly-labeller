#include "./aabb2d.h"
#include <algorithm>

namespace Math
{
Aabb2d::Aabb2d(Eigen::Vector2f center, Eigen::Vector2f halfWidthExtents)
  : center(center), halfWidthExtents(halfWidthExtents)
{
}

bool Aabb2d::testAabb(Aabb2d other)
{
  if (std::abs(center.x() - other.center.x()) >
      (halfWidthExtents.x() + other.halfWidthExtents.x()))
    return false;
  if (std::abs(center.y() - other.center.y()) >
      (halfWidthExtents.y() + other.halfWidthExtents.y()))
    return false;

  return true;
}
}  // namespace Math
