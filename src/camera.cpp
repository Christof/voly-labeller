#include "./camera.h"
#include <Eigen/Geometry>
#include <math.h>

#include <iostream>

Camera::Camera()
  : origin(0, 0, 0), position(0, 0, -1), direction(0, 0, 1), up(0, 1, 0),
    radius(1.0f), azimuth(-M_PI / 2.0f), declination(0)
{
  projection = createProjection(fieldOfView, aspectRatio, near, far);
  // projection = createOrthographicProjection(aspectRatio, near, far);

  update();
}

Camera::Camera(Eigen::Matrix4f viewMatrix, Eigen::Matrix4f projectionMatrix,
               Eigen::Vector3f origin)
  : projection(projectionMatrix), view(viewMatrix), origin(origin)
{
  position = -viewMatrix.inverse().col(3).head<3>();
  direction = viewMatrix.col(2).head<3>();
  up = viewMatrix.col(1).head<3>();

  radius = (position - origin).norm();

  Eigen::Vector3f diff = (position - origin) / radius;
  declination = asin(diff.y());
  azimuth = acos(diff.x() / cos(declination));
}

Camera::~Camera()
{
}

Eigen::Matrix4f Camera::createProjection(float fov, float aspectRatio,
                                         float nearPlane, float farPlane)
{
  double tanHalfFovy = tan(fov / 2.0);
  Eigen::Matrix4f result = Eigen::Matrix4f::Zero();
  result(0, 0) = 1.0 / (aspectRatio * tanHalfFovy);
  result(1, 1) = 1.0 / (tanHalfFovy);
  result(2, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
  result(3, 2) = -1.0;
  result(2, 3) = -(2.0 * farPlane * nearPlane) / (farPlane - nearPlane);

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

void Camera::update()
{
  radius = (origin - position).norm();
  position = origin +
             Eigen::Vector3f(cos(azimuth) * cos(declination), sin(declination),
                             sin(azimuth) * cos(declination)) *
                 radius;
  direction = (origin - position).normalized();
  float upDeclination = declination - M_PI / 2.0f;
  up = -Eigen::Vector3f(cos(azimuth) * cos(upDeclination), sin(upDeclination),
                        sin(azimuth) * cos(upDeclination)).normalized();

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

void Camera::updateNearAndFarPlanes(float near, float far)
{
  this->near = near;
  this->far = far;

  projection = createProjection(fieldOfView, aspectRatio, near, far);
}

bool Camera::needsResizing()
{
  return aspectRatio == 0.0f;
}

