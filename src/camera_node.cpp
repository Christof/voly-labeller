#include "./camera_node.h"
#include <vector>
#include <string>
#include "./graphics/gl.h"
#include "./graphics/managers.h"

CameraNode::CameraNode()
{
  camera = std::make_shared<Camera>();
}

CameraNode::CameraNode(std::shared_ptr<Camera> camera,
                       std::vector<CameraPosition> cameraPositions)
  : cameraPositions(cameraPositions), camera(camera)
{
}

void CameraNode::render(Graphics::Gl *gl,
                        std::shared_ptr<Graphics::Managers> managers,
                        RenderData renderData)
{
}

std::shared_ptr<Camera> CameraNode::getCamera()
{
  return camera;
}
void CameraNode::setOnCameraPositionsChanged(
    std::function<void(std::vector<CameraPosition>)> onChanged)
{
  onCameraPositionsChanged = onChanged;

  onCameraPositionsChanged(cameraPositions);
}

void CameraNode::saveCameraPosition(std::string name,
                                    Eigen::Matrix4f viewMatrix)
{
  cameraPositions.push_back(CameraPosition(name, viewMatrix));

  if (onCameraPositionsChanged)
    onCameraPositionsChanged(cameraPositions);
}

void CameraNode::removeCameraPosition(int index)
{
  cameraPositions.erase(cameraPositions.begin() + index);

  if (onCameraPositionsChanged)
    onCameraPositionsChanged(cameraPositions);
}

