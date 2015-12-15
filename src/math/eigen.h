#ifndef SRC_MATH_EIGEN_H_

#define SRC_MATH_EIGEN_H_

#include <Eigen/Core>
#include <QPoint>

inline Eigen::Vector3f toVector3f(const Eigen::Vector4f vector)
{
  return Eigen::Vector3f(vector.x(), vector.y(), vector.z());
}

inline Eigen::Vector4f toVector4f(const Eigen::Vector3f vector)
{
  return Eigen::Vector4f(vector.x(), vector.y(), vector.z(), 1.0f);
}

inline Eigen::Vector4f mul(const Eigen::Matrix4f &matrix,
                           const Eigen::Vector3f &vector)
{
  return matrix * toVector4f(vector);
}

inline Eigen::Vector2f toEigen(const QPoint &pos)
{
  return Eigen::Vector2f(pos.x(), pos.y());
}

inline Eigen::Vector2f toEigen(const QPointF &pos)
{
  return Eigen::Vector2f(pos.x(), pos.y());
}

inline Eigen::Vector2f ndcToPixels(Eigen::Vector2f ndc, Eigen::Vector2f size)
{
  return (ndc * 0.5f + Eigen::Vector2f(0.5f, 0.5f)).cwiseProduct(size);
}

#endif  // SRC_MATH_EIGEN_H_
