#include "./coordinate_system_node.h"
#include <Eigen/Core>

CoordinateSystemNode::CoordinateSystemNode()
{
  persistable = false;

  x = std::make_shared<Graphics::Connector>(Eigen::Vector3f(0, 0, 0),
                                            Eigen::Vector3f(1, 0, 0));
  x->color = Eigen::Vector4f(1, 0, 0, 1);

  y = std::make_shared<Graphics::Connector>(Eigen::Vector3f(0, 0, 0),
                                            Eigen::Vector3f(0, 1, 0));
  y->color = Eigen::Vector4f(0, 1, 0, 1);

  z = std::make_shared<Graphics::Connector>(Eigen::Vector3f(0, 0, 0),
                                            Eigen::Vector3f(0, 0, 1));
  z->color = Eigen::Vector4f(0, 0, 1, 1);

  obb = Math::Obb(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 1, 1),
                  Eigen::Matrix3f::Identity());
}

CoordinateSystemNode::~CoordinateSystemNode()
{
}

void CoordinateSystemNode::render(Graphics::Gl *gl,
                                  std::shared_ptr<Graphics::Managers> managers,
                                  RenderData renderData)
{
  if (!isVisible)
    return;

  x->render(gl, managers, renderData);
  y->render(gl, managers, renderData);
  z->render(gl, managers, renderData);
}

