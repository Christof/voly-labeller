#ifndef SRC_CAMERA_H_

#define SRC_CAMERA_H_

#include <Eigen/Core>
#include <Eigen/Geometry>

/**
 * \brief Camera which can be moved around and uses perspective projection.
 *
 *
 */
class Camera
{
 public:
  Camera();
  Camera(Eigen::Matrix4f viewMatrix, Eigen::Matrix4f projectionMatrix,
         Eigen::Vector3f origin);
  virtual ~Camera();

  Eigen::Matrix4f getProjectionMatrix() const;
  Eigen::Matrix4f getViewMatrix() const;
  Eigen::Vector3f getPosition() const;
  Eigen::Vector3f getOrigin() const;
  float getRadius() const;

  void moveForward(float distance);
  void moveBackward(float distance);
  void strafeLeft(float distance);
  void strafeRight(float distance);
  void strafe(float distance);
  void moveVertical(float distance);

  void changeAzimuth(float deltaAngle);
  void changeDeclination(float deltaAngle);
  void changeRadius(float deltaRadius);

  void resize(float width, float height);

  void updateNearAndFarPlanes(float nearPlane, float farPlane);

  float getNearPlane();
  float getFarPlane();

  void setOrigin(Eigen::Vector3f origin);

  bool needsResizing();

  void startAnimation(Eigen::Matrix4f viewMatrix, float duration);

  void updateAnimation(double frameTime);

 private:
  float nearPlane = 0.1f;
  float farPlane = 5.0f;
  Eigen::Matrix4f projection;
  Eigen::Matrix4f view;

  Eigen::Vector3f origin;
  Eigen::Vector3f position;
  Eigen::Vector3f direction;
  Eigen::Vector3f up;

  float radius;
  float azimuth;
  float declination;

  float fieldOfView = static_cast<float>(0.25 * M_PI);
  float aspectRatio = 0.0f;

  float animationTime = 0.0f;
  float animationDuration = 0.0f;
  Eigen::Vector3f animationEndPosition;
  Eigen::Vector3f animationStartPosition;
  Eigen::Quaternionf animationStartRotation;
  Eigen::Quaternionf animationEndRotation;

  Eigen::Matrix4f createProjection(float fov, float aspectRatio,
                                   float nearPlane, float farPlane);
  Eigen::Matrix4f createOrthographicProjection(float aspectRatio,
                                               float nearPlane, float farPlane);
  void update();
};

#endif  // SRC_CAMERA_H_
