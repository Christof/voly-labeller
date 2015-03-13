#include "./camera.h"
#include <Eigen/Geometry>
#include <iostream>
#include <math.h>

Camera::Camera()
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

Eigen::Matrix4f Camera::getViewMatrix()
{
  return view;
}

Eigen::Matrix4f Camera::getProjectionMatrix()
{
  return projection;
}
