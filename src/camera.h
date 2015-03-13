#ifndef SRC_CAMERA_H_

#define SRC_CAMERA_H_

#include <Eigen/Core>

/**
 * \brief Camera which can be moved around and uses perspective projection.
 *
 *
 */
class Camera
{
 public:
  Camera();
  virtual ~Camera();

  Eigen::Matrix4f getProjectionMatrix();
  Eigen::Matrix4f getViewMatrix();

 private:
  Eigen::Matrix4f projection;
  Eigen::Matrix4f view;

  Eigen::Matrix4f createProjection(float fov, float aspectRatio, float nearPlane, float farPlane);
};

#endif  // SRC_CAMERA_H_
