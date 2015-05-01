#ifndef SRC_EIGEN_QDEBUG_H_

#define SRC_EIGEN_QDEBUG_H_

#include <Eigen/Core>
#include <QDebug>
#include <iostream>

QDebug operator<<(QDebug dbg, const Eigen::Vector2f &vector)
{
  dbg << "[" << vector.x() << "|" << vector.y() << "]";

  return dbg.maybeSpace();
}

QDebug operator<<(QDebug dbg, const Eigen::Vector3f &vector)
{
  dbg << "[" << vector.x() << "|" << vector.y() << "|" << vector.z() << "]";

  return dbg.maybeSpace();
}

QDebug operator<<(QDebug dbg, const Eigen::Vector4f &vector)
{
  dbg << "[" << vector.x() << "|" << vector.y() << "|" << vector.z() << "|"
      << vector.w() << "]";

  return dbg.maybeSpace();
}

QDebug operator<<(QDebug dbg, const Eigen::Matrix4f &vector)
{
  const Eigen::IOFormat cleanFormat(4, 0, ", ", "\n", "[", "]");
  std::stringstream stream;
  stream << vector.format(cleanFormat);
  dbg << stream.str().c_str();

  return dbg.maybeSpace();
}

#endif  // SRC_EIGEN_QDEBUG_H_
