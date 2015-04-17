#include "./coordinate_system_node.h"
#include <Eigen/Core>

CoordinateSystemNode::CoordinateSystemNode()
{
  x = std::make_shared<Connector>(Eigen::Vector3f(0, 0, 0),
                                  Eigen::Vector3f(1, 0, 0));
  x->color = Eigen::Vector4f(1, 0, 0, 1);

  y = std::make_shared<Connector>(Eigen::Vector3f(0, 0, 0),
                                  Eigen::Vector3f(0, 1, 0));
  y->color = Eigen::Vector4f(0, 1, 0, 1);

  z = std::make_shared<Connector>(Eigen::Vector3f(0, 0, 0),
                                  Eigen::Vector3f(0, 0, 1));
  z->color = Eigen::Vector4f(0, 0, 1, 1);
}

CoordinateSystemNode::~CoordinateSystemNode()
{
}

void CoordinateSystemNode::render(Gl *gl, RenderData renderData)
{
  x->render(gl, renderData);
  y->render(gl, renderData);
  z->render(gl, renderData);
}

