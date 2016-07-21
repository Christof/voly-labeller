#include "./render_data.h"

RenderData::RenderData()
  : frameTime(0), projectionMatrix(Eigen::Matrix4f::Identity()),
    viewMatrix(Eigen::Matrix4f::Identity()), cameraPosition(0, 0, 0),
    modelMatrix(Eigen::Matrix4f::Identity()), windowPixelSize(0, 0),
    viewProjectionMatrix(Eigen::Matrix4f::Identity())
{
}

RenderData::RenderData(double frameTime, Eigen::Matrix4f projectionMatrix,
                       Eigen::Matrix4f viewMatrix,
                       Eigen::Vector3f cameraPosition,
                       Eigen::Vector2f windowPixelSize)
  : frameTime(frameTime), projectionMatrix(projectionMatrix),
    viewMatrix(viewMatrix), cameraPosition(cameraPosition),
    modelMatrix(Eigen::Matrix4f::Identity()), windowPixelSize(windowPixelSize),
    viewProjectionMatrix(projectionMatrix * viewMatrix)
{
}
