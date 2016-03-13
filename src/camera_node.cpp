#include "./camera_node.h"
#include "./graphics/gl.h"
#include "./graphics/managers.h"

CameraNode::CameraNode()
{
  camera = std::make_shared<Camera>();
}

CameraNode::CameraNode(std::shared_ptr<Camera> camera)
  : camera(camera)
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
