#include "./camera.h"

Camera::Camera()
{
}

Camera::~Camera()
{
}

Eigen::Matrix4f Camera::GetViewMatrix()
{
  return Eigen::Matrix4f::Identity();
}

Eigen::Matrix4f Camera::GetProjectionMatrix()
{
  return Eigen::Matrix4f::Identity();
}
