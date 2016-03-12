#include "./render_data.h"

RenderData::RenderData(Eigen::Matrix4f projectionMatrix,
                       Eigen::Matrix4f viewMatrix,
                       Eigen::Vector3f cameraPosition,
                       Eigen::Vector2f windowPixelSize)
  : projectionMatrix(projectionMatrix), viewMatrix(viewMatrix),
    cameraPosition(cameraPosition), modelMatrix(Eigen::Matrix4f::Identity()),
    windowPixelSize(windowPixelSize)
{
}
