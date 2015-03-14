#include "./camera.h"
#include <Eigen/Geometry>
#include <iostream>
#include <math.h>

Camera::Camera() : position(1, 0, 4), direction(0, 0, -1), up(0, 1, 0)
{
  projection = createProjection(M_PI / 2.0f, 16.0f / 9.0f, 0.1f, 100.0f);

  Eigen::Affine3f transformation(Eigen::Translation3f(0, 0, -4));
  view = transformation.matrix();
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

void Camera::strafeLeft(float distance)
{
  auto right = direction.cross(up);
  position -= distance * right;
}

void Camera::strafeRight(float distance)
{
  auto right = direction.cross(up);
  position += distance * right;
}

Eigen::Matrix4f Camera::getViewMatrix()
{
  std::cout << "pos: " << position << std::endl;
  auto n = direction.normalized();
  auto u = up.cross(n).normalized();
  auto v = n.cross(u);
  auto e = position;

  view << u.x(), u.y(), u.z(), u.dot(e),
       v.x(), v.y(), v.z(), v.dot(e),
       n.x(), n.y(), n.z(), n.dot(e),
       0, 0, 0, 1;

  std::cout << view << std::endl;

  return view;
}

Eigen::Matrix4f Camera::getProjectionMatrix()
{
  return projection;
}

