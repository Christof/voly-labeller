#include "./camera.h"
#include <Eigen/Geometry>
#include <math.h>

Camera::Camera()
  : position(0, 0, -1.0f), direction(0, 0, 1), up(0, 1, 0), radius(1.0f),
    azimuth(-M_PI / 2.0f), declination(0)
{
  projection = createProjection(M_PI / 2.0f, 16.0f / 9.0f, 0.1f, 100.0f);
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

void Camera::moveForward(float distance)
{
  position += distance * direction;
}

void Camera::moveBackward(float distance)
{
  position -= distance * direction;
}

void Camera::strafe(float distance)
{
  auto right = direction.cross(up);
  position += distance * right;
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
  position = position.normalized() * radius;
  update();
}

void Camera::update()
{
  radius = position.norm();
  position = Eigen::Vector3f(cos(azimuth) * cos(declination), sin(declination),
                             sin(azimuth) * cos(declination)) *
             radius;
  direction = -position.normalized();
  float upDeclination = declination - M_PI / 2.0f;
  up = -Eigen::Vector3f(cos(azimuth) * cos(upDeclination), sin(upDeclination),
                        sin(azimuth) * cos(upDeclination)).normalized();
}

Eigen::Matrix4f Camera::getViewMatrix()
{
  auto n = direction.normalized();
  auto u = up.cross(n).normalized();
  auto v = n.cross(u);
  auto e = position;

  view << u.x(), u.y(), u.z(), u.dot(e),
       v.x(), v.y(), v.z(), v.dot(e),
       n.x(), n.y(), n.z(), n.dot(e),
       0, 0, 0, 1;

  return view;
}

Eigen::Matrix4f Camera::getProjectionMatrix()
{
  return projection;
}

Eigen::Vector3f Camera::getPosition()
{
  return position;
}

