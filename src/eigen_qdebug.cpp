#include "./eigen_qdebug.h"

QDebug operator<<(QDebug dbg, const Eigen::Vector2f &vector)
{
  dbg << "[" << vector.x() << "|" << vector.y() << "]";

  return dbg.maybeSpace();
}

QDebug operator<<(QDebug dbg, const Eigen::Vector2i &vector)
{
  dbg << "[" << vector.x() << "|" << vector.y() << "]";

  return dbg.maybeSpace();
}

QDebug operator<<(QDebug dbg, const Eigen::Vector3f &vector)
{
  dbg << "[" << vector.x() << "|" << vector.y() << "|" << vector.z() << "]";

  return dbg.maybeSpace();
}

QDebug operator<<(QDebug dbg, const Eigen::Vector3i &vector)
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

QDebug operator<<(QDebug dbg, const Eigen::Matrix4f &matrix)
{
  const Eigen::IOFormat cleanFormat(4, 0, ", ", "\n", "\t[", "]");
  std::stringstream stream;
  stream << matrix.format(cleanFormat);
  dbg << "\n" << stream.str().c_str();

  return dbg.maybeSpace();
}

QDebug operator<<(QDebug dbg, const Eigen::Matrix3f &matrix)
{
  const Eigen::IOFormat cleanFormat(4, 0, ", ", "\n", "\t[", "]");
  std::stringstream stream;
  stream << matrix.format(cleanFormat);
  dbg << "\n" << stream.str().c_str();

  return dbg.maybeSpace();
}

