#ifndef SRC_MATH_AABB2D_H_

#define SRC_MATH_AABB2D_H_

#include <Eigen/Core>

namespace Math
{
/**
 * \brief Axis-aligned bounding box in 2D
 *
 * The bounding box is represented by its center and the half width
 * extents.
 */
class Aabb2d
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Aabb2d(Eigen::Vector2f center, Eigen::Vector2f halfWidthExtents);

  bool testAabb(Aabb2d other);

 private:
  Eigen::Vector2f center;
  Eigen::Vector2f halfWidthExtents;
};
}  // namespace Math

#endif  // SRC_MATH_AABB2D_H_
