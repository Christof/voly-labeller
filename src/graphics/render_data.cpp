#include "./render_data.h"

RenderData::RenderData()
  : projectionMatrix(Eigen::Matrix4f::Identity()),
    viewMatrix(Eigen::Matrix4f::Identity()), cameraPosition(0, 0, 0),
    modelMatrix(Eigen::Matrix4f::Identity()), windowPixelSize(0, 0),
    viewProjectionMatrix(Eigen::Matrix4f::Identity())
{
}

RenderData::RenderData(Eigen::Matrix4f projectionMatrix,
                       Eigen::Matrix4f viewMatrix,
                       Eigen::Vector3f cameraPosition,
                       Eigen::Vector2f windowPixelSize)
  : projectionMatrix(projectionMatrix), viewMatrix(viewMatrix),
    cameraPosition(cameraPosition), modelMatrix(Eigen::Matrix4f::Identity()),
    windowPixelSize(windowPixelSize),
    viewProjectionMatrix(projectionMatrix * viewMatrix)
{
}
