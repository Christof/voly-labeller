#ifndef SRC_MATH_EIGEN_H_

#define SRC_MATH_EIGEN_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
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

inline Eigen::Vector3f project(const Eigen::Matrix4f &matrix,
                               const Eigen::Vector4f &vector)
{
  Eigen::Vector4f result = matrix * vector;
  return result.head<3>() / result.w();
}

inline Eigen::Vector3f project(const Eigen::Matrix4f &matrix,
                               const Eigen::Vector3f &vector)
{
  return project(matrix, toVector4f(vector));
}

inline Eigen::Vector3f calculateWorldScale(Eigen::Vector4f sizeNDC,
                                    Eigen::Matrix4f projectionMatrix)
{
  Eigen::Vector3f sizeWorld = project(projectionMatrix.inverse(),
      Eigen::Vector4f(sizeNDC.x(), sizeNDC.y(), sizeNDC.z(), 1));

  return Eigen::Vector3f(sizeWorld.x(), sizeWorld.y(), 1.0f);
}

inline Eigen::Vector2f toPixel(Eigen::Vector2f ndc, Eigen::Vector2f size)
{
  const Eigen::Vector2f half(0.5f, 0.5f);

  auto zeroToOne = ndc.head<2>().cwiseProduct(half) + half;

  return zeroToOne.cwiseProduct(size);
}

inline Eigen::Vector2f toPixel(Eigen::Vector2f ndc, Eigen::Vector2i size)
{
  return toPixel(ndc, static_cast<Eigen::Vector2f>(size.cast<float>()));
}

#endif  // SRC_MATH_EIGEN_H_
