#ifndef SRC_EIGEN_QDEBUG_H_

#define SRC_EIGEN_QDEBUG_H_

#include <Eigen/Core>
#include <QDebug>
#include <iostream>

QDebug operator<<(QDebug dbg, const Eigen::Vector2f &vector);

QDebug operator<<(QDebug dbg, const Eigen::Vector3f &vector);

QDebug operator<<(QDebug dbg, const Eigen::Vector4f &vector);

QDebug operator<<(QDebug dbg, const Eigen::Matrix4f &vector);

#endif  // SRC_EIGEN_QDEBUG_H_
