#ifndef SRC_CAMERA_H_

#define SRC_CAMERA_H_

#include <Eigen/Core>

/**
 * \brief Camera which can be moved around and uses perspective projeciton.
 *
 *
 */
class Camera
{
 public:
  Camera();
  virtual ~Camera();

  Eigen::Matrix4f GetProjectionMatrix();
  Eigen::Matrix4f GetViewMatrix();

 private:
  /* data */
};

#endif  // SRC_CAMERA_H_
