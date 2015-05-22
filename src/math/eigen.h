#ifndef SRC_MATH_EIGEN_H_

#define SRC_MATH_EIGEN_H_

#include <Eigen/Core>
#include <QPoint>

inline Eigen::Vector4f mul(const Eigen::Matrix4f &matrix,
                           const Eigen::Vector3f &vector)
{
  Eigen::Vector4f operand(vector.x(), vector.y(), vector.z(), 1.0f);
  return matrix * operand;
}

inline Eigen::Vector2f toEigen(const QPoint &pos)
{
  return Eigen::Vector2f(pos.x(), pos.y());
}

#endif  // SRC_MATH_EIGEN_H_
