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
  Eigen::Vector3f getPosition();

 public:
  void moveForward(float distance);
  void moveBackward(float distance);
  void strafeLeft(float distance);
  void strafeRight(float distance);
  void strafe(float distance);
  void moveVertical(float distance);

  void changeAzimuth(float deltaAngle);
  void changeDeclination(float deltaAngle);
  void changeRadius(float deltaRadius);

 private:
  Eigen::Matrix4f projection;
  Eigen::Matrix4f view;

  Eigen::Vector3f position;
  Eigen::Vector3f direction;
  Eigen::Vector3f up;

  float radius;
  float azimuth;
  float declination;

  Eigen::Matrix4f createProjection(float fov, float aspectRatio,
                                   float nearPlane, float farPlane);
  void update();
};

#endif  // SRC_CAMERA_H_
