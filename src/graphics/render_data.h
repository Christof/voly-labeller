#ifndef SRC_GRAPHICS_RENDER_DATA_H_

#define SRC_GRAPHICS_RENDER_DATA_H_

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

  RenderData();

  RenderData(Eigen::Matrix4f projectionMatrix, Eigen::Matrix4f viewMatrix,
             Eigen::Vector3f cameraPosition, Eigen::Vector2f windowPixelSize);

  Eigen::Matrix4f projectionMatrix;
  Eigen::Matrix4f viewMatrix;
  Eigen::Vector3f cameraPosition;
  Eigen::Matrix4f modelMatrix;
  Eigen::Vector2f windowPixelSize;

  Eigen::Matrix4f viewProjectionMatrix;
};

#endif  // SRC_GRAPHICS_RENDER_DATA_H_
