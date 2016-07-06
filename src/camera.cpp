#include "./camera.h"
#include <Eigen/Geometry>
#include <math.h>
#include "./math/eigen.h"

#include <iostream>

Camera::Camera()
  : origin(0.0f, 0.0f, 0.0f), position(0.0f, 0.0f, -1.0f),
    direction(0.0f, 0.0f, 1.0f), up(0.0f, 1.0f, 0.0f), radius(1.0f),
    azimuth(static_cast<float>(-M_PI / 2.0)), declination(0.0f)
{
  projection = createProjection(fieldOfView, aspectRatio, nearPlane, farPlane);
  // projection = createOrthographicProjection(aspectRatio, nearPlane,
  // farPlane);

  update();
}

Camera::Camera(Eigen::Matrix4f viewMatrix, Eigen::Matrix4f projectionMatrix,
               Eigen::Vector3f origin)
  : projection(projectionMatrix), view(viewMatrix), origin(-origin)
{
  position = viewMatrix.inverse().col(3).head<3>();
  direction = viewMatrix.col(2).head<3>();
  up = viewMatrix.col(1).head<3>();

  radius = (position - origin).norm();

  Eigen::Vector3f diff = (position - origin) / radius;
  declination = asin(diff.y());
  azimuth = -acos(diff.x() / cos(declination));
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
  radius += deltaRadius;
  position = origin + (position - origin).normalized() * radius;
  update();
}

void Camera::rotateAroundOrbit(Eigen::Vector2f delta)
{
  float normSquared = delta.dot(delta);
  Eigen::Vector3f to;
  if (normSquared > 1.0)
  {
    to = 1.0f / sqrt(normSquared) * Eigen::Vector3f(-delta.x(), delta.y(), 0);
  }
  else
  {
    to = Eigen::Vector3f(-delta.x(), delta.y(), sqrt(1 - normSquared));
  }

  Eigen::Quaternionf rotation;
  rotation.setFromTwoVectors(Eigen::Vector3f(0, 0, 1), to);
  // Eigen::Vector3f(delta.x(), delta.y(), 1).normalized());

  Eigen::Matrix3f rotationMatrix = rotation.matrix();
  /*
  Eigen::Matrix3f rotationMatrix;
  direction.normalize();
  auto right = up.cross(direction).normalized();
  rotationMatrix =
      Eigen::AngleAxisf(-delta.y(), right) * Eigen::AngleAxisf(-delta.x(), up);
      */

  // direction = rotationMatrix * direction;
  // up = rotationMatrix * up;

  Eigen::Vector3f originToCamera = position - origin;
  /*
  Eigen::Matrix4f moveToCamera =
      Eigen::Affine3f(Eigen::Translation3f(originToCamera)).matrix();

  Eigen::Affine3f transformation(Eigen::Translation3f(positionToOrigin) *
                                 rotationMatrix *
                                 Eigen::Translation3f(-positionToOrigin));
                                 */
  Eigen::Matrix4f rotMat = Eigen::Matrix4f::Identity();
  rotMat.block<3, 3>(0, 0) = rotationMatrix;
  position = toVector3f(mul(rotMat, originToCamera)) + origin;

  direction = (origin - position).normalized();
  auto right = up.cross(direction).normalized();
  up = direction.cross(right).normalized();
  // Eigen::Vector3f dirFromPos = (origin - position).normalized();
  std::cout << "Dir: " << (direction) << std::endl;
  // update();

  view << right.x(), right.y(), right.z(), right.dot(position), up.x(), up.y(),
      up.z(), up.dot(position), direction.x(), direction.y(), direction.z(),
      direction.dot(position), 0, 0, 0, 1;
}

void Camera::update()
{
  auto n = direction.normalized();
  auto u = up.cross(n).normalized();
  auto v = n.cross(u);
  auto e = position;

  view << u.x(), u.y(), u.z(), u.dot(e), v.x(), v.y(), v.z(), v.dot(e), n.x(),
      n.y(), n.z(), n.dot(e), 0, 0, 0, 1;
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
  return -origin;
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

void Camera::setOrigin(Eigen::Vector3f origin)
{
  this->origin = -origin;
  Eigen::Vector3f diff = (position - this->origin).normalized();

  declination = asin(diff.y());
  azimuth = -acos(diff.x() / cos(declination));
  update();
}

bool Camera::needsResizing()
{
  return aspectRatio == 0.0f;
}

