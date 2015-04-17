#ifndef SRC_RENDER_DATA_H_

#define SRC_RENDER_DATA_H_

#include <Eigen/Core>

/**
 * \brief
 *
 *
 */
struct RenderData
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix4f projectionMatrix;
  Eigen::Matrix4f viewMatrix;
  Eigen::Matrix4f modelMatrix;
  Eigen::Vector3f cameraPosition;
};

#endif  // SRC_RENDER_DATA_H_
