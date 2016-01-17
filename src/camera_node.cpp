#include "./camera_node.h"

CameraNode::CameraNode(Camera camera) : camera(camera)
{
}

void CameraNode::render(Graphics::Gl *gl,
                        std::shared_ptr<Graphics::Managers> managers,
                        RenderData renderData)
{
}

