#ifndef SRC_COLLISION_H_

#define SRC_COLLISION_H_

#include "./eigen.h"

/**
 * \brief Returns 2 times the signed area of the triangle
 *
 * The result is positive if the triangle is counter-clock-wise
 * and negative if it is clock-wise. If the triangle is degenerate
 * the result is 0.
 *
 * Implementation taken from real-time collision detection by Christer Ericson
 */
float signed2DTriangelArea(Eigen::Vector2f a, Eigen::Vector2f b,
                           Eigen::Vector2f c)
{
  return (a.x() - c.x()) * (b.y() - c.y()) - (a.y() - c.y()) * (b.x() - c.x());
}

/**
 * \brief Test if line segments ab and cd overlap.
 */
bool test2DSegmentSegment(Eigen::Vector2f a, Eigen::Vector2f b,
                         Eigen::Vector2f c, Eigen::Vector2f d)
{
  // Sign of areas correspond to which side of ab points c and d are
  float area1 = signed2DTriangelArea(a, b, d);
  float area2 = signed2DTriangelArea(a, b, c);

  // If c and d are on same side of ab, areas have same signs
  // which means the product is positive and there is no collision.
  if (area1 * area2 >= 0.0f)
    return false;

  // Compute signs for a and b witch respect to segment cd
  float area3 = signed2DTriangelArea(c, d, a);
  // Since area is constant a1 - a2 = a3 - a4 so a4 can be calculated as
  float area4 = area3 - area1 + area2;

  return area3 * area4 < 0.0f;
}

#endif  // SRC_COLLISION_H_
