#include "./camera.h"
#include <Eigen/Geometry>
#include <math.h>

Camera::Camera()
  : origin(0.0f, 0.0f, 0.0f), position(0.0f, 0.0f, 1.0f),
    direction(0.0f, 0.0f, -1.0f), up(0.0f, 1.0f, 0.0f)
{
  projection = createProjection(fieldOfView, aspectRatio, nearPlane, farPlane);
  // projection = createOrthographicProjection(aspectRatio, nearPlane,
  // farPlane);

  setOrigin(origin);
}

Camera::Camera(Eigen::Matrix4f viewMatrix, Eigen::Matrix4f projectionMatrix,
               Eigen::Vector3f origin)
  : projection(projectionMatrix), view(viewMatrix), origin(origin)
{
  setPosDirUpFrom(viewMatrix);

  setOrigin(origin);
}

Camera::~Camera()
{
}

Eigen::Matrix4f Camera::createProjection(float fov, float aspectRatio,
                                         float nearPlane, float farPlane)
{
  float tanHalfFovy = tan(fov / 2.0f);
  Eigen::Matrix4f result = Eigen::Matrix4f::Zero();
  result(0, 0) = 1.0f / (aspectRatio * tanHalfFovy);
  result(1, 1) = 1.0f / (tanHalfFovy);
  result(2, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
  result(3, 2) = -1.0f;
  result(2, 3) = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);

  return result;
}

Eigen::Matrix4f Camera::createOrthographicProjection(float aspectRatio,
                                                     float nearPlane,
                                                     float farPlane)
{
  float diff = farPlane - nearPlane;
  Eigen::Matrix4f result = Eigen::Matrix4f::Zero();
  result(0, 0) = 2.0f;
  result(1, 1) = 2.0f * aspectRatio;
  result(2, 2) = 1.0f / diff;
  result(2, 3) = -nearPlane / diff;
  result(3, 3) = 1.0f;

  return result;
}

void Camera::moveForward(float distance)
{
  position += distance * direction;
  update();
}

void Camera::moveBackward(float distance)
{
  position -= distance * direction;
  update();
}

void Camera::strafe(float distance)
{
  auto right = direction.cross(up);
  position += distance * right;
  origin += distance * right;
  update();
}

void Camera::strafeLeft(float distance)
{
  strafe(-distance);
}

void Camera::strafeRight(float distance)
{
  strafe(distance);
}

void Camera::moveVertical(float distance)
{
  position += distance * up;
  origin += distance * up;
  update();
}

void Camera::changeAzimuth(float deltaAngle)
{
  azimuth += deltaAngle;
  update();
}

void Camera::changeDeclination(float deltaAngle)
{
  declination += deltaAngle;
  update();
}

void Camera::changeRadius(float deltaRadius)
{
  auto radius = getRadius() + deltaRadius;
  position = origin - getLookAt() * radius;
  update();
}

Eigen::Vector3f unitVectorFromAngles(float azimuth, float declination)
{
  return Eigen::Vector3f(sin(azimuth) * sin(declination), cos(declination),
                         cos(azimuth) * sin(declination));
}

void Camera::update()
{
  auto radius = getRadius();
  position = origin + unitVectorFromAngles(azimuth, declination) * radius;

  float upDeclination = declination - 0.5f * static_cast<float>(M_PI);
  up = unitVectorFromAngles(azimuth, upDeclination).normalized();

  direction = getLookAt();
  auto zAxis = -direction;
  auto xAxis = up.cross(zAxis).normalized();
  up = zAxis.cross(xAxis);

  view << xAxis.x(), xAxis.y(), xAxis.z(), -xAxis.dot(position), up.x(), up.y(),
      up.z(), -up.dot(position), zAxis.x(), zAxis.y(), zAxis.z(),
      -zAxis.dot(position), 0, 0, 0, 1;
}

Eigen::Matrix4f Camera::getViewMatrix() const
{
  return view;
}

Eigen::Matrix4f Camera::getProjectionMatrix() const
{
  return projection;
}

Eigen::Vector3f Camera::getPosition() const
{
  return position;
}

Eigen::Vector3f Camera::getOrigin() const
{
  return origin;
}

float Camera::getRadius() const
{
  return (position - origin).norm();
}

void Camera::resize(float width, float height)
{
  aspectRatio = width / height;
  projection = createProjection(fieldOfView, aspectRatio, 0.1f, 100.0f);
}

void Camera::updateNearAndFarPlanes(float nearPlane, float farPlane)
{
  this->nearPlane = nearPlane;
  this->farPlane = farPlane;

  projection = createProjection(fieldOfView, aspectRatio, nearPlane, farPlane);
}

float Camera::getNearPlane()
{
  return this->nearPlane;
}

float Camera::getFarPlane()
{
  return this->farPlane;
}

Eigen::Vector3f Camera::getLookAt() const
{
  return (origin - position).normalized();
}

void Camera::setOrigin(Eigen::Vector3f origin)
{
  this->origin = origin;
  Eigen::Vector3f originToPositionNormalized = -getLookAt();

  setAnglesFromUnitVector(originToPositionNormalized);
  update();
}

bool Camera::needsResizing()
{
  return aspectRatio == 0.0f;
}

void Camera::startAnimation(Eigen::Matrix4f viewMatrix, float duration)
{
  animationTime = 0.0f;
  animationDuration = duration;

  animationStartPosition = position;

  animationStartRotation = view.block<3, 3>(0, 0).inverse();
  animationEndRotation = viewMatrix.block<3, 3>(0, 0).inverse();
  animationEndPosition = viewMatrix.inverse().col(3).head<3>();
}

void Camera::updateAnimation(double frameTime)
{
  if (animationDuration == 0.0f)
    return;

  animationTime += frameTime;
  float factor = animationTime / animationDuration;
  if (animationTime > animationDuration)
  {
    factor = 1;
    animationDuration = 0.0f;
  }

  // factor = easeInOutQuad(factor);

  Eigen::Vector3f diff =
      factor * (animationEndPosition - animationStartPosition);

  position = animationStartPosition + diff;

  auto rotation = animationStartRotation.slerp(factor, animationEndRotation);
  Eigen::Matrix3f rotationMatrix = rotation.toRotationMatrix();

  direction = -rotationMatrix.col(2).normalized();
  up = rotationMatrix.col(1).normalized();
  // origin = position + direction;  // originDiff.norm() * direction;
  setOrigin(position + direction);

  /*
  setAnglesFromUnitVector(-direction);

  update();
  //setOrigin(origin);
  */
}

void Camera::setPosDirUpFrom(Eigen::Matrix4f viewMatrix)
{
  position = viewMatrix.inverse().col(3).head<3>();
  direction = -viewMatrix.col(2).head<3>();
  up = viewMatrix.col(1).head<3>();
}

void Camera::setAnglesFromUnitVector(Eigen::Vector3f unitVectorInSphere)
{
  declination = acos(unitVectorInSphere.y());
  azimuth = atan2(unitVectorInSphere.x(), unitVectorInSphere.z());
}

