#ifndef SRC_RENDER_DATA_H_

#define SRC_RENDER_DATA_H_

#include <Eigen/Core>

/**
 * \brief Encapsulates data which is necessary to render objects
 *
 * It includes for now only camera data.
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
